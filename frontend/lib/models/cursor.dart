// Cursor 데이터 모델
// (음표 하나하나에 대응하는 커서 정보 저장)
class Cursor {
  final double x; // 전체 캔버스 기준 X(px)
  final double y; // 전체 캔버스 기준 Y(px)
  final double w; // 커서 가로 크기
  final double h; // 커서 세로 크기
  final double ts; // 음표 timestamp (시간 정보, 박자 기준)
  final double? xRatio; // (상대좌표) x 좌표 비율 (이미지 너비에 대한 비율, null이면 x 좌표 사용)
  final double? yRatio; // (상대좌표) 악보 줄 내부 세로 비율
  final int measureNumber; // OSMD CurrentMeasureIndex
  final int lineIndex; // 시스템 (줄) 번호

  Cursor({
    required this.x,
    required this.y,
    required this.w,
    required this.h,
    required this.ts,
    this.xRatio,
    this.yRatio,
    required this.measureNumber,
    required this.lineIndex,
  });

  // 비어있는 기본 커서 생성자
  Cursor.createEmpty()
      : x = 0,
        y = 0,
        w = 0,
        h = 0,
        ts = 0,
        xRatio = null,
        yRatio = null,
        measureNumber = -1,
        lineIndex = -1;

  // 복사본 생성 (일부 값만 바꿔서 새 객체 만들기)
  Cursor copyWith({
    double? x,
    double? y,
    double? w,
    double? h,
    double? ts,
    double? xRatio,
    double? yRatio,
    int? measureNumber,
    int? lineIndex,
  }) {
    return Cursor(
      x: x ?? this.x,
      y: y ?? this.y,
      w: w ?? this.w,
      h: h ?? this.h,
      ts: ts ?? this.ts,
      xRatio: xRatio ?? this.xRatio,
      yRatio: yRatio ?? this.yRatio,
      measureNumber: measureNumber ?? this.measureNumber,
      lineIndex: lineIndex ?? this.lineIndex,
    );
  }

  // JSON → Cursor 변환
  factory Cursor.fromJson(Map<String, dynamic> json) {
    return Cursor(
      x: (json['x'] as num).toDouble(),
      y: (json['y'] as num).toDouble(),
      w: (json['w'] as num).toDouble(),
      h: (json['h'] as num).toDouble(),
      ts: (json['ts'] as num).toDouble(),
      xRatio:
          json['xRatio'] != null ? (json['xRatio'] as num).toDouble() : null,
      yRatio:
          json['yRatio'] != null ? (json['yRatio'] as num).toDouble() : null,
      measureNumber: (json['measureNumber'] as num?)?.toInt() ?? -1,
      lineIndex: (json['lineIndex'] as num?)?.toInt() ?? -1,
    );
  }

  // Cursor → JSON 변환
  Map<String, dynamic> toJson() {
    return {
      'x': x,
      'y': y,
      'w': w,
      'h': h,
      'ts': ts,
      if (xRatio != null) 'xRatio': xRatio,
      if (yRatio != null) 'yRatio': yRatio,
      'measureNumber': measureNumber,
      'lineIndex': lineIndex,
    };
  }
}
