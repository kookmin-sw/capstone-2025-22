// Cursor 데이터 모델
// (음표 하나하나에 대응하는 커서 정보 저장)
class Cursor {
  final double x; // x 좌표 (캔버스 or 이미지 기준)
  final double y; // y 좌표
  final double w; // 커서 가로 크기
  final double h; // 커서 세로 크기
  final double ts; // 음표 timestamp (시간 정보, 박자 기준)
  final double? xRatio; // (선택) x 좌표 비율 (이미지 너비에 대한 비율, null이면 x 좌표 사용)

  Cursor({
    required this.x,
    required this.y,
    required this.w,
    required this.h,
    required this.ts,
    this.xRatio,
  });

  // 비어있는 기본 커서 생성자
  Cursor.createEmpty()
      : x = 0,
        y = 0,
        w = 0,
        h = 0,
        ts = 0,
        xRatio = null;

  // 복사본 생성 (일부 값만 바꿔서 새 객체 만들기)
  Cursor copyWith({
    double? x,
    double? y,
    double? w,
    double? h,
    double? ts,
    num? xRatio,
  }) {
    return Cursor(
      x: x ?? this.x,
      y: y ?? this.y,
      w: w ?? this.w,
      h: h ?? this.h,
      ts: ts ?? this.ts,
      xRatio: (xRatio ?? this.xRatio)?.toDouble(),
    );
  }

  // (선택) JSON → Cursor 변환
  factory Cursor.fromJson(Map<String, dynamic> json) {
    return Cursor(
      x: (json['x'] as num).toDouble(),
      y: (json['y'] as num).toDouble(),
      w: (json['w'] as num).toDouble(),
      h: (json['h'] as num).toDouble(),
      ts: (json['ts'] as num).toDouble(),
      xRatio:
          json['xRatio'] != null ? (json['xRatio'] as num).toDouble() : null,
    );
  }

  // (선택) Cursor → JSON 변환
  Map<String, dynamic> toJson() {
    return {
      'x': x,
      'y': y,
      'w': w,
      'h': h,
      'ts': ts,
      if (xRatio != null) 'xRatio': xRatio,
    };
  }
}
