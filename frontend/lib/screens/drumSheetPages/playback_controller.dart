import 'dart:async';
import 'dart:typed_data';
import 'package:capstone_2025/models/sheet_info.dart';
import 'package:flutter/foundation.dart';
import '../../models/cursor.dart';
import './cursor_controller.dart';

class PlaybackController {
  CursorController? _cursorController; // 커서 이동 관리
  double? canvasWidth; // 커서 위치 계산용 캔버스 원본 너비
  SheetInfo? sheetInfo;

  // 재생 상태 및 타이머
  double speed = 1.0; // 재생 속도 (배속)
  bool isPlaying = false; // 현재 재생 중인지 여부
  bool isCountingDown = false; // 카운트다운 중인지 여부
  int countdown = 3; // 카운트다운 초 수
  DateTime? playbackStartTime; // 재생 시작 시간
  Timer? countdownTimer; // 카운트다운용 타이머
  Timer? progressTimer; // 재생 중 진행 관리 타이머

  // 재생 진행도
  Duration totalDuration = Duration.zero; // 전체 재생 시간
  Duration currentDuration = Duration.zero; // 현재 재생 시간
  double currentProgress = 0.0; // 전체 대비 현재 진행 비율 (0.0 ~ 1.0)

  // 페이지 / 줄 이동 관리
  int currentPage = 0; // 현재 재생 중인 줄 인덱스

  // 커서 데이터
  List<Cursor> fullCursorList = []; // 전체 커서 리스트 (진행 시간 계산용)
  Cursor currentCursor = Cursor.createEmpty();

  // 줄별 악보 이미지 관련
  List<Uint8List> lineImages = []; // 줄 단위로 잘라낸 악보 이미지들
  Uint8List? currentLineImage; // 현재 줄의 악보 이미지 (줄 넘어갈 때마다 바뀜)
  Uint8List? nextLineImage; // 다음 줄 미리보기 악보 이미지
  final double imageHeight; // 이미지 높이 저장

  // 콜백 함수들
  Function(double)? onProgressUpdate;
  Function(bool)? onPlaybackStateChange;
  Function(int)? onCountdownUpdate;
  Function(int)? onPageChange;

  PlaybackController({required this.imageHeight}); // 생성자에 imageHeight 추가

  void loadSheetInfo(SheetInfo? info) {
    if (info == null) return;
    sheetInfo = info;
    fullCursorList = sheetInfo!.cursorList;
    print('📊 Loaded full cursor list: ${fullCursorList.length} cursors');

    lineImages = sheetInfo!.lineImages;

    // ➊ 전체 리스트로 한 번만 컨트롤러 생성
    _cursorController?.dispose();
    _cursorController = CursorController(
      cursorList: fullCursorList,
      bpm: sheetInfo!.bpm.toDouble(),
      speed: speed,
      onCursorMove: _handleCursorMove,
    );

    currentPage = 0;
    currentLineImage = lineImages.isNotEmpty ? lineImages[0] : null;
    nextLineImage = lineImages.length > 1 ? lineImages[1] : null;
  }

  /// 커서가 이동할 때마다 호출됩니다.
  void _handleCursorMove(Cursor cursor) {
    // 1) 위치 업데이트
    updateCursorWidget(cursor);

    // 2) 줄이 바뀌면 이미지 교체
    if (cursor.lineIndex != currentPage) {
      currentPage = cursor.lineIndex;
      currentLineImage = lineImages[currentPage];
      nextLineImage = (currentPage + 1 < lineImages.length)
          ? lineImages[currentPage + 1]
          : null;
      onPageChange?.call(currentPage);
    }
  }

  void updateCursorWidget(Cursor cursor) {
    if (canvasWidth == null || lineImages.isEmpty) return;

    double adjustedX;
    if (cursor.xRatio != null) {
      adjustedX = cursor.xRatio! * canvasWidth!;
    } else {
      adjustedX = cursor.x;
    }

    // y 좌표도 함께 업데이트
    double adjustedY = cursor.y;
    if (cursor.yRatio != null) {
      // 전달받은 imageHeight를 사용하여 y 좌표 계산
      adjustedY = cursor.yRatio! * imageHeight;
    }

    final adjustedCursor = cursor.copyWith(
      x: adjustedX,
      y: adjustedY,
    );
    currentCursor = adjustedCursor;
  }

  // 전체 재생 시간 세팅 (진행바 계산, 재생 완료 판별용)
  void setTotalDuration(Duration duration) {
    totalDuration = duration;
  }

  void startPlayback() async {
    if (lineImages.isEmpty) return; // 줄 이미지가 없는 경우 방어

    // 현재 재생 상태를 유지하기 위해 실제 재생 시작된 시간을 기록
    playbackStartTime = DateTime.now().subtract(Duration(
      milliseconds: (currentDuration.inMilliseconds / speed).round(),
    ));

    // 진행 업데이트를 위한 타이머 재설정 (기존 타이머 중지 후 새로 시작, 재생 속도 반영)
    progressTimer?.cancel();
    progressTimer = Timer.periodic(
        Duration(milliseconds: (100 ~/ speed).round()), _onProgressTick);

    isPlaying = true;
    onPlaybackStateChange?.call(isPlaying);

    // 처음 시작하는 건지 재시작인지 구분
    if (currentDuration == Duration.zero) {
      // 새로 시작
      _cursorController?.start();
    } else {
      // 재개(resume) 부분 직전에,
      // 멈춘 순간의 커서 모양을 즉시 화면에 반영
      final cur = _cursorController?.getCurrentCursor();
      if (cur != null) updateCursorWidget(cur);
      _cursorController?.resume();
    }
  }

  // Timer 콜백 : 재생 시간 업데이트 + 줄 이동 관리
  void _onProgressTick(Timer timer) async {
    if (!isPlaying || playbackStartTime == null) {
      timer.cancel(); // 재생 중 아니거나 시작시간 없으면 타이머 종료
      return;
    }

    try {
      final now = DateTime.now();
      final elapsed = now.difference(playbackStartTime!);

      // 현재 재생 시간 업데이트 (speed 반영)
      int newMs = (elapsed.inMilliseconds * speed).round();
      if (newMs > totalDuration.inMilliseconds) {
        newMs = totalDuration.inMilliseconds;
      }
      currentDuration = Duration(milliseconds: newMs);

      // 진행 퍼센트 업데이트 콜백 호출
      if (totalDuration.inMilliseconds > 0) {
        currentProgress =
            currentDuration.inMilliseconds / totalDuration.inMilliseconds;
        onProgressUpdate?.call(currentProgress);
      }

      // 전체 재생 완료 여부 체크
      if (currentDuration >= totalDuration) {
        stopPlayback();
        return;
      }
    } catch (e) {
      debugPrint("Error in _onProgressTick: $e");
      timer.cancel();
      stopPlayback();
    }
  }

  void stopPlayback() {
    progressTimer?.cancel(); // 재생 진행 타이머 중지
    _cursorController?.stop(); // 커서 이동 타이머 중지
    isPlaying = false;
    onPlaybackStateChange?.call(isPlaying);
  }

  void resetToStart() {
    // 1) 타이머 & 컨트롤러 모두 중지
    stopPlayback();
    // 2) 진행 상태만 리셋
    currentDuration = Duration.zero;
    currentProgress = 0.0;
    // 3) 첫 페이지 & 이미지로 초기화
    currentPage = 0;
    if (lineImages.isNotEmpty) {
      currentLineImage = lineImages[0];
      nextLineImage = lineImages.length > 1 ? lineImages[1] : null;
    }

    // 4) 콜백으로 화면 갱신
    onProgressUpdate?.call(currentProgress); // 진행바 0으로 초기화
    onPageChange?.call(currentPage); // 화면 줄 이동 콜백

    // 5) 커서 컨트롤러 위치만 초기화 (재시작은 하지 않음)
    _cursorController?.stop();
    if (fullCursorList.isNotEmpty) {
      updateCursorWidget(fullCursorList.first);
    }
  }

  void showCountdownAndStart() {
    _cursorController?.stop();

    isCountingDown = true;
    countdown = 3;
    onCountdownUpdate?.call(countdown);

    countdownTimer?.cancel();
    countdownTimer = Timer.periodic(const Duration(seconds: 1), (timer) {
      countdown--;
      onCountdownUpdate?.call(countdown);

      if (countdown <= 0) {
        timer.cancel();
        isCountingDown = false;
        _onCountdownComplete();
      }
    });
  }

  void calculateTotalDurationFromCursorList(double bpm) {
    if (fullCursorList.isEmpty) {
      debugPrint("❗ 커서 리스트가 비어 있음. 재생 시간 계산 생략");
      totalDuration = Duration.zero;
      return;
    }

    // 혹시 모를 ts 오름차순 정렬
    fullCursorList.sort((a, b) => a.ts.compareTo(b.ts));

    final firstTS = fullCursorList.first.ts;
    final lastTS = fullCursorList.last.ts;

    if (firstTS == lastTS || lastTS < firstTS) {
      debugPrint("❗ 타임스탬프가 이상함. 재생 시간 계산 생략");
      totalDuration = Duration.zero;
      return;
    }

    final secondsPerBeat = 60 / bpm; // 1박자당 걸리는 초 계산
    final durationInBeats =
        (lastTS - firstTS) + speed; // 시작 커서와 끝 커서 박자 차이 계산 + 1박자 여유 두기
    final durationMs =
        (durationInBeats * secondsPerBeat * 1000).toInt(); // 전체 곡 재생 시간 계산
    totalDuration = Duration(milliseconds: durationMs);

    debugPrint("⏱️ 총 재생 시간(ms): $durationMs");
  }

  void _onCountdownComplete() {
    startPlayback(); // 카운트다운 끝나면 재생 시작
  }

  void setSpeed(double newSpeed) {
    // 1) speed 값만 업데이트
    speed = newSpeed;
    _cursorController?.setSpeed(newSpeed);

    // 2) UI 리빌드용 콜백 (선택)
    onPlaybackStateChange?.call(isPlaying);
  }

  void dispose() {
    countdownTimer?.cancel();
    progressTimer?.cancel();
    _cursorController?.dispose();
  }
}
