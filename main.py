from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import google.generativeai as genai
import os, json, base64, math
import numpy as np
import cv2
from PIL import Image
import io

app = FastAPI(
    title="CogniScan — Clock Drawing Service",
    description="Test 2: Visuospatial Assessment — Feature extraction via OpenCV, scoring via Gemini",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")


# ─── Models ───────────────────────────────────────────────────────────────────

class ClockSubmitRequest(BaseModel):
    patient_id: str
    age: int
    education_years: int
    image_base64: str          # PNG/JPG drawing from Flutter canvas
    time_taken_seconds: int
    stroke_count: int
    hesitation_pauses: int     # pauses > 2s while drawing
    target_time: str = "10:10" # what time they were asked to draw

class ClockFeatures(BaseModel):
    # Circle
    circle_detected: bool
    circle_completeness: float     # 0-1
    circle_regularity: float       # how round it is, 0-1

    # Numbers
    numbers_found: int             # 0-12
    numbers_in_correct_positions: int
    numbers_outside_circle: int
    number_spacing_regularity: float  # 0-1, even spacing = 1

    # Hands
    hand_count: int                # 0, 1, or 2
    hands_correct_length_ratio: bool  # minute hand longer than hour hand
    hands_pointing_correct_time: bool

    # Process metrics
    time_taken_seconds: int
    stroke_count: int
    hesitation_pauses: int

class ClockDrawingResult(BaseModel):
    patient_id: str
    visuospatial_score: float      # 0-10, for aggregator
    clock_score: float             # 0-4, clinical CDT scale
    features: dict
    domain_breakdown: dict         # circle/numbers/hands/hands_time each scored
    clinical_flags: List[str]
    interpretation: str
    raw_data: dict


# ─── Feature Extraction (OpenCV) — NO tokens used ─────────────────────────────

def extract_clock_features(image_base64: str, stroke_count: int,
                            hesitation_pauses: int, time_taken: int) -> dict:
    """
    Extract all measurable geometric features from the clock drawing.
    This runs entirely locally — zero API tokens consumed.
    """
    # Decode image
    img_bytes = base64.b64decode(image_base64)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image data")

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ── Preprocessing ──────────────────────────────────────────
    # Invert if white-on-black (Flutter canvas default is black on white)
    mean_brightness = np.mean(gray)
    if mean_brightness < 128:
        gray = cv2.bitwise_not(gray)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # ── 1. Circle Detection ────────────────────────────────────
    circle_data = _detect_circle(gray, binary, h, w)

    # ── 2. Number Detection ────────────────────────────────────
    number_data = _detect_numbers(binary, circle_data, h, w)

    # ── 3. Hand Detection ─────────────────────────────────────
    hand_data = _detect_hands(binary, circle_data, h, w)

    return {
        # Circle
        "circle_detected": circle_data["detected"],
        "circle_completeness": circle_data["completeness"],
        "circle_regularity": circle_data["regularity"],
        "circle_center": circle_data["center"],
        "circle_radius": circle_data["radius"],

        # Numbers
        "numbers_found": number_data["count"],
        "numbers_in_correct_positions": number_data["correct_positions"],
        "numbers_outside_circle": number_data["outside_circle"],
        "number_spacing_regularity": number_data["spacing_regularity"],
        "number_quadrant_distribution": number_data["quadrant_distribution"],

        # Hands
        "hand_count": hand_data["count"],
        "hands_correct_length_ratio": hand_data["correct_length_ratio"],
        "hands_pointing_correct_time": hand_data["correct_time"],
        "hand_angle_difference": hand_data["angle_difference"],

        # Process
        "time_taken_seconds": time_taken,
        "stroke_count": stroke_count,
        "hesitation_pauses": hesitation_pauses,
        "drawing_density": float(np.sum(binary > 0)) / (h * w),
        "image_dimensions": [w, h],
    }


def _detect_circle(gray, binary, h, w):
    """Detect the outer clock circle using Hough Circles."""
    min_radius = int(min(h, w) * 0.25)
    max_radius = int(min(h, w) * 0.50)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=min(h, w) // 2,
        param1=50,
        param2=30,
        minRadius=min_radius,
        maxRadius=max_radius
    )

    if circles is not None:
        c = np.round(circles[0][0]).astype(int)
        cx, cy, r = int(c[0]), int(c[1]), int(c[2])

        # Measure completeness — what % of expected circle has ink
        mask = np.zeros_like(gray)
        cv2.circle(mask, (cx, cy), r, 255, 8)
        circle_pixels = np.sum(mask > 0)
        actual_ink = np.sum((binary > 0) & (mask > 0))
        completeness = min(1.0, float(actual_ink) / max(circle_pixels, 1))

        # Regularity — ratio of circle area to convex hull area of all strokes
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            all_points = np.vstack(contours)
            hull = cv2.convexHull(all_points)
            hull_area = cv2.contourArea(hull)
            circle_area = math.pi * r * r
            regularity = min(1.0, circle_area / max(hull_area, 1))
        else:
            regularity = completeness

        return {
            "detected": True,
            "center": [cx, cy],
            "radius": r,
            "completeness": round(completeness, 2),
            "regularity": round(regularity, 2)
        }
    else:
        # No circle detected — try to estimate from drawing bounds
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            all_points = np.vstack(contours)
            (cx, cy), r = cv2.minEnclosingCircle(all_points)
            return {
                "detected": False,
                "center": [int(cx), int(cy)],
                "radius": int(r),
                "completeness": 0.2,
                "regularity": 0.2
            }
        return {"detected": False, "center": [w//2, h//2], "radius": min(h,w)//3,
                "completeness": 0.0, "regularity": 0.0}


def _detect_numbers(binary, circle_data, h, w):
    """
    Estimate number count by detecting digit-like blobs inside the clock face.
    Not OCR — counts isolated ink regions at expected number positions.
    """
    cx, cy = circle_data["center"]
    r = circle_data["radius"]

    # Look for blobs at the 12 expected clock number positions
    correct_positions = 0
    outside_circle = 0
    found_positions = []

    for hour in range(1, 13):
        # Expected angle for each hour (12 o'clock = -90 degrees from center)
        angle_deg = (hour * 30) - 90
        angle_rad = math.radians(angle_deg)

        # Expected position at ~85% of radius
        expected_x = int(cx + r * 0.82 * math.cos(angle_rad))
        expected_y = int(cy + r * 0.82 * math.sin(angle_rad))

        # Check if there's ink in a small region around expected position
        margin = max(15, int(r * 0.12))
        x1 = max(0, expected_x - margin)
        x2 = min(w, expected_x + margin)
        y1 = max(0, expected_y - margin)
        y2 = min(h, expected_y + margin)

        region = binary[y1:y2, x1:x2]
        ink_density = np.sum(region > 0) / max(region.size, 1)

        if ink_density > 0.03:  # some ink present
            correct_positions += 1
            found_positions.append(hour)

    # Count blobs outside the circle
    mask_outside = np.zeros_like(binary)
    cv2.circle(mask_outside, (cx, cy), int(r * 1.1), 255, -1)
    outside_ink = np.sum((binary > 0) & (mask_outside == 0))
    outside_circle = 1 if outside_ink > (w * h * 0.002) else 0

    # Spacing regularity — are found numbers evenly distributed?
    if len(found_positions) >= 4:
        gaps = []
        for i in range(len(found_positions)):
            gap = found_positions[(i+1) % len(found_positions)] - found_positions[i]
            gaps.append(abs(gap))
        gap_std = np.std(gaps)
        spacing_regularity = max(0.0, 1.0 - (gap_std / 6.0))
    else:
        spacing_regularity = len(found_positions) / 12.0

    # Quadrant distribution (should have ~3 numbers per quadrant)
    quadrant_counts = [0, 0, 0, 0]
    for pos in found_positions:
        quadrant_counts[(pos - 1) // 3] += 1

    return {
        "count": correct_positions,
        "correct_positions": correct_positions,
        "outside_circle": outside_circle,
        "spacing_regularity": round(spacing_regularity, 2),
        "quadrant_distribution": quadrant_counts
    }


def _detect_hands(binary, circle_data, h, w):
    """Detect clock hands as line segments from center."""
    cx, cy = circle_data["center"]
    r = circle_data["radius"]

    # Mask to inner 90% of circle (ignore numbers area)
    inner_mask = np.zeros_like(binary)
    cv2.circle(inner_mask, (cx, cy), int(r * 0.85), 255, -1)
    inner_binary = cv2.bitwise_and(binary, inner_mask)

    # Detect lines using HoughLinesP
    lines = cv2.HoughLinesP(
        inner_binary,
        rho=1,
        theta=np.pi/180,
        threshold=20,
        minLineLength=int(r * 0.25),
        maxLineGap=int(r * 0.1)
    )

    if lines is None:
        return {"count": 0, "correct_length_ratio": False,
                "correct_time": False, "angle_difference": 0}

    # Filter lines that pass near center
    center_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # Distance from center to line
        dist = abs((y2-y1)*cx - (x2-x1)*cy + x2*y1 - y2*x1) / max(
            math.sqrt((y2-y1)**2 + (x2-x1)**2), 1)
        if dist < r * 0.25:
            length = math.sqrt((x2-x1)**2 + (y2-y1)**2)
            angle = math.degrees(math.atan2(y2-y1, x2-x1))
            center_lines.append({
                "length": length,
                "angle": angle % 360,
                "coords": [x1, y1, x2, y2]
            })

    # Sort by length, take top 2 as hands
    center_lines.sort(key=lambda l: l["length"], reverse=True)
    hands = center_lines[:2]

    count = len(hands)
    correct_length_ratio = False
    correct_time = False
    angle_difference = 0

    if count == 2:
        # Minute hand should be longer than hour hand
        correct_length_ratio = hands[0]["length"] > hands[1]["length"] * 1.1

        # For 10:10: hour hand ~300° (pointing to 10), minute hand ~60° (pointing to 2)
        # Angle difference between hands should be ~120°
        angle_difference = abs(hands[0]["angle"] - hands[1]["angle"])
        if angle_difference > 180:
            angle_difference = 360 - angle_difference

        # 10:10 → hands are ~120° apart
        correct_time = 100 <= angle_difference <= 140

    return {
        "count": count,
        "correct_length_ratio": correct_length_ratio,
        "correct_time": correct_time,
        "angle_difference": round(angle_difference, 1)
    }


# ─── Gemini Scoring — receives tiny JSON, not image ───────────────────────────

def score_with_gemini(features: dict, age: int, education_years: int) -> dict:
    """
    Send only extracted features (tiny JSON ~50 tokens) to Gemini.
    Never sends the image. Solves the token limit problem entirely.
    """
    prompt = f"""
You are a neuropsychologist scoring a Clock Drawing Test (CDT) for dementia screening.

Patient: Age {age}, Education {education_years} years.

Extracted geometric features (from image analysis):
- Circle detected: {features['circle_detected']}
- Circle completeness: {features['circle_completeness']} (0-1)
- Circle regularity: {features['circle_regularity']} (0-1)
- Numbers found at correct positions: {features['numbers_found']}/12
- Numbers outside circle: {features['numbers_outside_circle']}
- Number spacing regularity: {features['number_spacing_regularity']} (0-1)
- Clock hands detected: {features['hand_count']}
- Minute hand longer than hour hand: {features['hands_correct_length_ratio']}
- Hands pointing to correct time (10:10): {features['hands_pointing_correct_time']}
- Angle between hands: {features['hand_angle_difference']} degrees
- Time taken: {features['time_taken_seconds']} seconds
- Drawing hesitations: {features['hesitation_pauses']}

CDT Scoring criteria (Shulman 4-point scale):
4 = Perfect or near-perfect
3 = Minor visuospatial errors
2 = Inaccurate representation of time, visuospatial organization preserved
1 = Moderate visuospatial disorganization
0 = No reasonable representation of clock

Return ONLY this JSON (no markdown):
{{
  "clock_score": <0-4 on CDT scale>,
  "visuospatial_score": <0-10 normalized for age/education>,
  "domain_breakdown": {{
    "circle": <0-10>,
    "numbers": <0-10>,
    "hands_present": <0-10>,
    "hands_time": <0-10>
  }},
  "clinical_flags": [<list of concerns, empty if none>],
  "interpretation": "<2 sentences, plain English for family>",
  "doctor_note": "<clinical note for neurologist>"
}}
"""
    try:
        response = model.generate_content(prompt)
        raw = response.text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
    except Exception:
        return _fallback_scoring(features, age)


def _fallback_scoring(features: dict, age: int) -> dict:
    """Rule-based fallback if Gemini fails — app never crashes."""
    score = 0
    flags = []

    if features["circle_detected"] and features["circle_completeness"] > 0.6:
        score += 1
    else:
        flags.append("Circle incomplete or absent")

    if features["numbers_found"] >= 8:
        score += 1
    else:
        flags.append(f"Only {features['numbers_found']}/12 numbers detected")

    if features["hand_count"] >= 2:
        score += 1
    else:
        flags.append(f"Only {features['hand_count']} hand(s) detected")

    if features["hands_pointing_correct_time"]:
        score += 1
    else:
        flags.append("Hands not pointing to correct time (10:10)")

    visuospatial = (score / 4) * 10
    if age >= 70:
        visuospatial = min(10, visuospatial + 0.5)

    return {
        "clock_score": score,
        "visuospatial_score": round(visuospatial, 1),
        "domain_breakdown": {
            "circle": 10 if score >= 1 else 3,
            "numbers": min(10, features["numbers_found"] / 1.2),
            "hands_present": 10 if features["hand_count"] == 2 else 4,
            "hands_time": 10 if features["hands_pointing_correct_time"] else 2
        },
        "clinical_flags": flags,
        "interpretation": f"Clock drawing score: {score}/4. {'Some concerns noted.' if flags else 'Within normal range.'}",
        "doctor_note": f"CDT score {score}/4. Circle: {features['circle_detected']}, Numbers: {features['numbers_found']}/12, Hands: {features['hand_count']}."
    }


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "service": "CogniScan Clock Drawing",
        "status": "running",
        "test": "Test 2 — Visuospatial",
        "approach": "OpenCV feature extraction → tiny JSON → Gemini (no image tokens)"
    }


@app.post("/api/test/clock-drawing/submit", response_model=ClockDrawingResult)
def submit_clock(req: ClockSubmitRequest):
    """
    Accepts a base64 image of the clock drawing.
    OpenCV extracts all features locally (no tokens).
    Only a small JSON of features is sent to Gemini for clinical interpretation.
    """
    # Step 1: Extract features locally with OpenCV — 0 tokens
    features = extract_clock_features(
        req.image_base64,
        req.stroke_count,
        req.hesitation_pauses,
        req.time_taken_seconds
    )

    # Step 2: Send only tiny feature JSON to Gemini — ~50 tokens
    gemini_result = score_with_gemini(features, req.age, req.education_years)

    return ClockDrawingResult(
        patient_id=req.patient_id,
        visuospatial_score=gemini_result.get("visuospatial_score", 5.0),
        clock_score=gemini_result.get("clock_score", 2.0),
        features=features,
        domain_breakdown=gemini_result.get("domain_breakdown", {}),
        clinical_flags=gemini_result.get("clinical_flags", []),
        interpretation=gemini_result.get("interpretation", "Results recorded."),
        raw_data={
            **features,
            "clock_score": gemini_result.get("clock_score"),
            "doctor_note": gemini_result.get("doctor_note", ""),
            "age": req.age,
            "education_years": req.education_years,
        }
    )
