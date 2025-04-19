// lib/screens/drumSheetPages/playback_controller.dart

import 'dart:async';
import 'package:flutter_inappwebview/flutter_inappwebview.dart';

class PlaybackController {
  InAppWebViewController? _webViewController;
  bool isWebViewReady = false;

  // ⚙️ 재생 관련 상태 변수들
  double speed = 1.0;
  double step = 0.0;
  bool isPlaying = false;
  bool isCountingDown = false;
  int countdown = 3;
  DateTime? playbackStartTime;

  // ⏱️ 타이머
  Timer? countdownTimer;
  Timer? progressTimer;

  // 📊 진행도 상태
  final Duration totalDuration = const Duration(seconds: 60);
  Duration currentDuration = Duration.zero;
  double currentProgress = 0.0;
  int totalLines = 1;
  int currentPage = 0;

  // 콜백 함수들
  Function(double)? onProgressUpdate;
  Function(bool)? onPlaybackStateChange;
  Function(int)? onCountdownUpdate;
  Function(int)? onPageChange;

  set webViewController(InAppWebViewController controller) {
    _webViewController = controller;
    isWebViewReady = true;
  }

  void startPlayback() async {
    if (_webViewController == null) return;

    // ✅ 현재 페이지 줄 먼저 로딩
    await _webViewController!.evaluateJavascript(
      source: 'loadLine($currentPage);',
    );

    final lineCountJs = await _webViewController!
        .evaluateJavascript(source: "getTotalLineCount();");
    totalLines = int.tryParse(lineCountJs.toString()) ?? 1;

    final jsResult = await _webViewController!.evaluateJavascript(
      source: "getNoteCountForLine($currentPage);",
    );
    print("🟡 getNoteCountForLine(${currentPage}) JS Result: $jsResult");
    final totalNotes = int.tryParse(jsResult.toString()) ?? 0;
    if (totalNotes == 0) {
      print("❌ No notes found on line $currentPage");
      return; // 재생 중단
    }
    step = 1.0 / totalNotes;

    playbackStartTime = DateTime.now().subtract(Duration(
      milliseconds: (currentDuration.inMilliseconds / speed).round(),
    ));
    final progressBarInterval = Duration(milliseconds: (100 ~/ speed).round());

    progressTimer?.cancel();
    progressTimer = Timer.periodic(progressBarInterval, (timer) async {
      if (!isPlaying ||
          playbackStartTime == null ||
          _webViewController == null) {
        timer.cancel();
        return;
      }

      final now = DateTime.now();
      final elapsed = now.difference(playbackStartTime!);
      currentDuration =
          Duration(milliseconds: (elapsed.inMilliseconds * speed).round());
      currentProgress += step;

      onProgressUpdate?.call(currentProgress);

      await _webViewController!.evaluateJavascript(
        source: "moveNextCursorStep();",
      );

      if (currentProgress >= 1.0) {
        await _goToNextPage();
      }
      if (currentDuration >= totalDuration) {
        stopPlayback();
      }
    });

    isPlaying = true;
    onPlaybackStateChange?.call(isPlaying);
  }

  Future<void> _goToNextPage() async {
    if (_webViewController == null) return;

    currentPage = (currentPage + 1) % totalLines;
    currentProgress = 0.0;
    onPageChange?.call(currentPage);

    final jsResult = await _webViewController!.evaluateJavascript(
      source: "getNoteCountForLine($currentPage);",
    );
    final totalNotes = int.tryParse(jsResult.toString()) ?? 0;
    if (totalNotes == 0) {
      print("❌ No notes on line $currentPage. Skipping page.");
      return;
    }

    step = 1.0 / totalNotes;

    await _webViewController!.evaluateJavascript(
      source: 'loadLine($currentPage);',
    );
  }

  void stopPlayback() {
    progressTimer?.cancel();
    isPlaying = false;
    onPlaybackStateChange?.call(isPlaying);
  }

  void resetToStart() {
    stopPlayback();
    currentDuration = Duration.zero;
    currentProgress = 0.0;
    currentPage = 0;
    onProgressUpdate?.call(currentProgress);
    onPageChange?.call(currentPage);
  }

  void showCountdownAndStart() {
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
        startPlayback();
      }
    });
  }

  void setSpeed(double newSpeed) {
    speed = newSpeed;
    if (isPlaying) {
      stopPlayback();
      showCountdownAndStart();
    }
  }

  void dispose() {
    countdownTimer?.cancel();
    progressTimer?.cancel();
  }
}
