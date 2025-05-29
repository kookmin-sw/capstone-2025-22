import 'dart:async';
import '../../models/cursor.dart';

// CursorController는 음표(ts 기반) 리스트를 받아서
// bpm, 배속(speed)을 기준으로 정확한 타이밍에 맞춰 커서를 이동시키는 클래스
class CursorController {
  final List<Cursor> cursorList; // 이동해야 할 커서 리스트
  final double bpm; // 악보의 BPM (템포)
  double speed; // 배속 (1.0 = 정상 속도, 2.0 = 2배속)

  CursorController({
    required this.cursorList,
    required this.bpm,
    this.speed = 1.0,
  });

  Cursor getAdjustedCursorAtBeat(double beatTs) {
    if (cursorList.isEmpty) return Cursor.createEmpty();
    if (beatTs >= cursorList.last.ts) return cursorList.last;

    for (int i = 0; i < cursorList.length - 1; i++) {
      if (cursorList[i].ts <= beatTs && beatTs < cursorList[i + 1].ts) {
        return cursorList[i];
      }
    }
    // 만약 못 찾으면 (이론상 발생X), 첫 커서 반환
    return cursorList.first;
  }
}
