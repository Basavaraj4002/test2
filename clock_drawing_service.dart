import 'dart:convert';
import 'dart:ui' as ui;
import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';
import 'package:http/http.dart' as http;

const String BASE_URL = 'https://cogni-clock-drawing.onrender.com';

// ─────────────────────────────────────────────────────────────
// CLOCK DRAWING CANVAS WIDGET
// Drop this into your test screen
// ─────────────────────────────────────────────────────────────

class ClockDrawingCanvas extends StatefulWidget {
  final Function(String base64Image, int strokeCount, int hesitations) onComplete;

  const ClockDrawingCanvas({Key? key, required this.onComplete}) : super(key: key);

  @override
  State<ClockDrawingCanvas> createState() => _ClockDrawingCanvasState();
}

class _ClockDrawingCanvasState extends State<ClockDrawingCanvas> {
  final GlobalKey _canvasKey = GlobalKey();
  final List<List<Offset?>> _strokes = [];
  List<Offset?> _currentStroke = [];
  int _strokeCount = 0;
  int _hesitationCount = 0;
  DateTime? _lastStrokeTime;
  DateTime _drawingStartTime = DateTime.now();

  void _onPanStart(DragStartDetails details) {
    // Detect hesitation — gap > 2s between strokes
    if (_lastStrokeTime != null) {
      final gap = DateTime.now().difference(_lastStrokeTime!).inSeconds;
      if (gap >= 2) _hesitationCount++;
    }
    _currentStroke = [details.localPosition];
    _strokeCount++;
    setState(() {});
  }

  void _onPanUpdate(DragUpdateDetails details) {
    setState(() => _currentStroke.add(details.localPosition));
  }

  void _onPanEnd(DragEndDetails details) {
    setState(() {
      _currentStroke.add(null);
      _strokes.add(List.from(_currentStroke));
      _currentStroke = [];
      _lastStrokeTime = DateTime.now();
    });
  }

  void _clear() => setState(() {
    _strokes.clear();
    _strokeCount = 0;
    _hesitationCount = 0;
  });

  Future<void> _submit() async {
    // Capture canvas as PNG
    final boundary = _canvasKey.currentContext!.findRenderObject()
        as RenderRepaintBoundary;
    final image = await boundary.toImage(pixelRatio: 2.0);
    final byteData = await image.toByteData(format: ui.ImageByteFormat.png);
    final base64Image = base64Encode(byteData!.buffer.asUint8List());

    widget.onComplete(base64Image, _strokeCount, _hesitationCount);
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Text(
          'Draw a clock showing 10:10',
          style: TextStyle(fontSize: 18, fontWeight: FontWeight.w600),
        ),
        const SizedBox(height: 8),
        Text(
          'Draw the clock face, numbers, and both hands',
          style: TextStyle(color: Colors.grey[600], fontSize: 14),
        ),
        const SizedBox(height: 16),
        RepaintBoundary(
          key: _canvasKey,
          child: GestureDetector(
            onPanStart: _onPanStart,
            onPanUpdate: _onPanUpdate,
            onPanEnd: _onPanEnd,
            child: Container(
              width: 320,
              height: 320,
              decoration: BoxDecoration(
                color: Colors.white,
                border: Border.all(color: Colors.grey.shade300, width: 2),
                borderRadius: BorderRadius.circular(12),
              ),
              child: CustomPaint(
                painter: _ClockPainter(
                  strokes: _strokes,
                  currentStroke: _currentStroke,
                ),
              ),
            ),
          ),
        ),
        const SizedBox(height: 16),
        Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            OutlinedButton.icon(
              onPressed: _clear,
              icon: Icon(Icons.refresh),
              label: Text('Clear'),
            ),
            const SizedBox(width: 16),
            ElevatedButton.icon(
              onPressed: _strokes.isNotEmpty ? _submit : null,
              icon: Icon(Icons.check),
              label: Text('Done'),
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.indigo,
                foregroundColor: Colors.white,
                padding: EdgeInsets.symmetric(horizontal: 24, vertical: 12),
              ),
            ),
          ],
        ),
      ],
    );
  }
}

class _ClockPainter extends CustomPainter {
  final List<List<Offset?>> strokes;
  final List<Offset?> currentStroke;

  _ClockPainter({required this.strokes, required this.currentStroke});

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = Colors.black
      ..strokeWidth = 3.0
      ..strokeCap = StrokeCap.round
      ..style = PaintingStyle.stroke;

    for (final stroke in [...strokes, currentStroke]) {
      for (int i = 0; i < stroke.length - 1; i++) {
        if (stroke[i] != null && stroke[i + 1] != null) {
          canvas.drawLine(stroke[i]!, stroke[i + 1]!, paint);
        }
      }
    }
  }

  @override
  bool shouldRepaint(_ClockPainter old) => true;
}


// ─────────────────────────────────────────────────────────────
// CLOCK DRAWING API SERVICE
// ─────────────────────────────────────────────────────────────

class ClockDrawingService {
  static Future<ClockDrawingResult> submit({
    required String patientId,
    required int age,
    required int educationYears,
    required String imageBase64,
    required int timeTakenSeconds,
    required int strokeCount,
    required int hesitationPauses,
  }) async {
    final response = await http.post(
      Uri.parse('$BASE_URL/api/test/clock-drawing/submit'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({
        'patient_id': patientId,
        'age': age,
        'education_years': educationYears,
        'image_base64': imageBase64,
        'time_taken_seconds': timeTakenSeconds,
        'stroke_count': strokeCount,
        'hesitation_pauses': hesitationPauses,
        'target_time': '10:10',
      }),
    );

    if (response.statusCode == 200) {
      return ClockDrawingResult.fromJson(jsonDecode(response.body));
    } else {
      throw Exception('Clock drawing service error: ${response.body}');
    }
  }
}

class ClockDrawingResult {
  final String patientId;
  final double visuospatialScore;  // 0-10 → send to aggregator
  final double clockScore;          // 0-15 CLOX scale
  final Map<String, dynamic> features;
  final Map<String, dynamic> domainBreakdown;
  final List<String> clinicalFlags;
  final String interpretation;
  final Map<String, dynamic> rawData;

  ClockDrawingResult({
    required this.patientId,
    required this.visuospatialScore,
    required this.clockScore,
    required this.features,
    required this.domainBreakdown,
    required this.clinicalFlags,
    required this.interpretation,
    required this.rawData,
  });

  factory ClockDrawingResult.fromJson(Map<String, dynamic> json) {
    return ClockDrawingResult(
      patientId: json['patient_id'],
      visuospatialScore: (json['visuospatial_score'] as num).toDouble(),
      clockScore: (json['clock_score'] as num).toDouble(),
      features: json['features'],
      domainBreakdown: json['domain_breakdown'],
      clinicalFlags: List<String>.from(json['clinical_flags']),
      interpretation: json['interpretation'],
      rawData: json['raw_data'],
    );
  }
}
