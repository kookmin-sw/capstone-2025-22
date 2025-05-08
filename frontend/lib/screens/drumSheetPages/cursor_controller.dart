import 'dart:async';
import '../../models/cursor.dart';

// CursorController는 음표(ts 기반) 리스트를 받아서
// bpm, 배속(speed)을 기준으로 정확한 타이밍에 맞춰 커서를 이동시키는 클래스
class CursorController {
  final List<Cursor> cursorList; // 이동해야 할 커서 리스트
  final void Function(Cursor) onCursorMove; // 커서가 이동할 때 호출할 콜백 함수
  final double bpm; // 악보의 BPM (템포)

  double speed; // 배속 (1.0 = 정상 속도, 2.0 = 2배속)

  Timer? _timer; // 커서 이동 예약용 타이머
  int _currentIndex = 0; // 현재 커서 인덱스

  CursorController({
    required this.cursorList,
    required this.onCursorMove,
    required this.bpm,
    this.speed = 1.0,
  });

  // 현재 커서 가져오기
  Cursor? getCurrentCursor() {
    if (_currentIndex >= 0 && _currentIndex < cursorList.length) {
      return cursorList[_currentIndex];
    }
    return null;
  }

  // 재생 시작 (처음부터)
  void start() {
    if (cursorList.isEmpty) return;
    _timer?.cancel();
    _currentIndex = 0;
    onCursorMove(cursorList[0]); // 첫 커서로 바로 이동
    _scheduleNextStep(); // 다음 커서 이동 예약 시작
  }

  //  다음 스텝 이동 예약
  void _scheduleNextStep() {
    if (_currentIndex + 1 >= cursorList.length) return; // 마지막 커서면 종료

    final prevTS = cursorList[_currentIndex].ts; // 현재 커서의 timestamp
    final nextTS = cursorList[_currentIndex + 1].ts; // 다음 커서의 timestamp
    final deltaBeats = nextTS - prevTS; // 두 커서 사이 박자 수 차이

    final msPerBeat = 60000 / bpm; // 1박자당 걸리는 시간 (ms)
    final delayMs = (deltaBeats * msPerBeat / speed).ceil();

    _timer = Timer(Duration(milliseconds: delayMs), () {
      _currentIndex++;
      onCursorMove(cursorList[_currentIndex]); // 커서 이동

      // 마지막 음표면 더 이상 예약하지 않고 종료
      if (_currentIndex < cursorList.length - 1) {
        _scheduleNextStep();
      }
    });
  }

  Cursor getCursorAtBeat(double beatTs) {
    // 범위 밖 처리
    if (beatTs <= cursorList.first.ts) return cursorList.first;
    if (beatTs >= cursorList.last.ts) return cursorList.last;

    // 이진 탐색으로 beatTs 에 가장 가까운 두 인덱스(lo, hi) 찾기
    int lo = 0, hi = cursorList.length - 1;
    while (hi - lo > 1) {
      final mid = (lo + hi) >> 1;
      if (cursorList[mid].ts <= beatTs) {
        lo = mid;
      } else {
        hi = mid;
      }
    }

    // lo, hi 두 포인트 사이를 비율 보간
    final a = cursorList[lo];
    final b = cursorList[hi];
    final segment = b.ts - a.ts;
    final t = segment > 0 ? (beatTs - a.ts) / segment : 0.0;
    return a.copyWith(
      x: a.x + (b.x - a.x) * t,
      y: a.y + (b.y - a.y) * t,
      xRatio: (a.xRatio != null && b.xRatio != null)
          ? a.xRatio! + (b.xRatio! - a.xRatio!) * t
          : null,
      yRatio: (a.yRatio != null && b.yRatio != null)
          ? a.yRatio! + (b.yRatio! - a.yRatio!) * t
          : null,
    );
  }

  //  일시정지
  void stop() {
    _timer?.cancel();
  }

  // 재시작
  void resume() {
    // 이미 currentIndex가 마지막으로 멈춘 상태이므로
    // 다음 스텝 예약만 해 주면 됨
    _timer?.cancel();
    _scheduleNextStep();
  }

  // 처음부터 재시작
  void reset() {
    stop();
    _currentIndex = 0; // 초기화
    start();
  }

  // 배속 변경 (중간에 변경해도 반영)
  void setSpeed(double newSpeed) {
    speed = newSpeed;
    if (_timer?.isActive ?? false) {
      _timer?.cancel();
      _scheduleNextStep(); // 새 배속에 맞춰 다시 예약
    }
  }

  // 리소스 해제
  void dispose() {
    _timer?.cancel();
  }
}
