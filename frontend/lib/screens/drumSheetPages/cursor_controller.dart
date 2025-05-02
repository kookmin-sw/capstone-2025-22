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
    final delayMs =
        (deltaBeats * msPerBeat / speed).round(); // 배속을 고려한 다음 이동까지 대기 시간

    _timer = Timer(Duration(milliseconds: delayMs), () {
      _currentIndex++;
      onCursorMove(cursorList[_currentIndex]); // 커서 이동
      _scheduleNextStep(); // 다시 다음 이동 예약
    });
  }

  //  일시정지
  void stop() {
    _timer?.cancel();
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
