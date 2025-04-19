// 🎵 drum_sheet_player.dart — 네가 만든 UI 그대로, 로직만 안정화

import 'dart:async';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_inappwebview/flutter_inappwebview.dart';
import 'playback_controller.dart';

class DrumSheetPlayer extends StatefulWidget {
  const DrumSheetPlayer({super.key});

  @override
  State<DrumSheetPlayer> createState() => _DrumSheetPlayerState();
}

class _DrumSheetPlayerState extends State<DrumSheetPlayer> {
  late InAppWebViewController webViewController;
  late PlaybackController playbackController;
  bool isWebViewReady = false;
  double imageHeight = 120;

  @override
  void initState() {
    super.initState();
    playbackController = PlaybackController()
      ..onProgressUpdate = (progress) {
        setState(() {});
      }
      ..onPlaybackStateChange = (isPlaying) {
        setState(() {});
      }
      ..onCountdownUpdate = (count) {
        setState(() {});
      }
      ..onPageChange = (page) {
        setState(() {});
      };
  }

  @override
  void dispose() {
    playbackController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final nextPage =
        (playbackController.currentPage + 1) % playbackController.totalLines;
    String sheetName = '그라데이션';
    String artistName = '10CM';

    return Scaffold(
      backgroundColor: const Color(0xFFF5F5F5),
      body: Stack(
        children: [
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 40),
            child: Column(
              children: [
                // 🎵 상단 컨트롤 바
                SizedBox(
                  height: 60,
                  child: Stack(
                    children: [
                      Row(
                        children: [
                          Expanded(
                            child: Row(
                              children: [
                                const SizedBox(width: 30),
                                const Icon(Icons.home,
                                    size: 30, color: Color(0xff646464)),
                                const SizedBox(width: 30),
                                Expanded(
                                  child: Container(
                                    constraints:
                                        const BoxConstraints(maxWidth: 400),
                                    padding: const EdgeInsets.symmetric(
                                        horizontal: 16, vertical: 12),
                                    decoration: BoxDecoration(
                                      color: Colors.white,
                                      borderRadius: BorderRadius.circular(18),
                                      border: Border.all(
                                          color: const Color(0xFFDFDFDF),
                                          width: 2),
                                    ),
                                    child: Text(
                                      '$sheetName - $artistName',
                                      overflow: TextOverflow.ellipsis,
                                      textAlign: TextAlign.center,
                                      style: const TextStyle(
                                          fontSize: 20, height: 1.2),
                                    ),
                                  ),
                                ),
                                const SizedBox(width: 100),
                              ],
                            ),
                          ),
                          Row(
                            children: [
                              const SizedBox(width: 100),
                              Container(
                                padding: const EdgeInsets.symmetric(
                                    horizontal: 23, vertical: 12),
                                decoration: BoxDecoration(
                                  color: Colors.white,
                                  borderRadius: BorderRadius.circular(18),
                                  border: Border.all(
                                      color: const Color(0xFFDFDFDF), width: 2),
                                ),
                                child: Row(
                                  mainAxisSize: MainAxisSize.min,
                                  children: [
                                    Padding(
                                      padding: const EdgeInsets.only(right: 20),
                                      child: GestureDetector(
                                        onTap: () =>
                                            playbackController.resetToStart(),
                                        child: const Icon(Icons.replay,
                                            size: 28, color: Color(0xff646464)),
                                      ),
                                    ),
                                    ...[0.5, 1.0, 1.5, 2.0].map((s) => Padding(
                                          padding: EdgeInsets.only(
                                              left: 15,
                                              right: s == 2.0 ? 0 : 15),
                                          child: GestureDetector(
                                            onTap: () {
                                              playbackController.setSpeed(s);
                                            },
                                            child: Text(
                                              '${s}x',
                                              style: TextStyle(
                                                fontSize: 20,
                                                fontWeight: FontWeight.bold,
                                                color: playbackController
                                                            .speed ==
                                                        s
                                                    ? const Color(0xffD97D6C)
                                                    : const Color(0xff646464),
                                              ),
                                            ),
                                          ),
                                        )),
                                  ],
                                ),
                              ),
                              const SizedBox(width: 40),
                            ],
                          ),
                        ],
                      ),
                      Center(
                        child: GestureDetector(
                          onTap: () => playbackController.isPlaying
                              ? playbackController.stopPlayback()
                              : playbackController.showCountdownAndStart(),
                          child: playbackController.isPlaying
                              ? Container(
                                  width: 52,
                                  height: 52,
                                  decoration: BoxDecoration(
                                    shape: BoxShape.circle,
                                    color: Colors.white,
                                    border: Border.all(
                                        color: const Color(0xFFDFDFDF),
                                        width: 2),
                                  ),
                                  child: const Icon(Icons.pause,
                                      size: 40, color: Color(0xffD97D6C)),
                                )
                              : Container(
                                  width: 52,
                                  height: 52,
                                  decoration: const BoxDecoration(
                                    shape: BoxShape.circle,
                                    color: Color(0xffD97D6C),
                                  ),
                                  child: const Icon(Icons.play_arrow,
                                      size: 40, color: Colors.white),
                                ),
                        ),
                      ),
                    ],
                  ),
                ),
                const SizedBox(height: 24),

                // 📄 악보 표시 영역
                Column(
                  children: [
                    Stack(
                      children: [
                        Container(
                          height: imageHeight,
                          width: double.infinity,
                          decoration: BoxDecoration(
                            color: Colors.white,
                            borderRadius: BorderRadius.circular(12),
                          ),
                          clipBehavior: Clip.hardEdge,
                          child: InAppWebView(
                            initialFile: 'assets/web/index.html',
                            onWebViewCreated: (controller) async {
                              webViewController = controller;
                              playbackController.webViewController = controller;
                              await InAppWebViewController
                                  .clearAllCache(); // 나중에 지우기
                              controller.addJavaScriptHandler(
                                handlerName: 'sendFileToOSMD',
                                callback: (args) async {
                                  String xml = await rootBundle
                                      .loadString('assets/music/demo.xml');
                                  return xml;
                                },
                              );
                              controller.addJavaScriptHandler(
                                handlerName: 'onCursorStep',
                                callback: (args) {
                                  final cursorData = args[0];
                                  setState(() {
                                    playbackController.currentProgress =
                                        cursorData['x'] ?? 0.0;
                                  });
                                  return null;
                                },
                              );
                            },
                            onLoadStop: (controller, url) async {
                              isWebViewReady = true;
                              print("✅ WebView load complete");
                              print(
                                  "📄 Loading line: ${playbackController.currentPage}");
                              await controller.evaluateJavascript(
                                source:
                                    'loadLine(${playbackController.currentPage});',
                              );
                              await controller.evaluateJavascript(
                                  source: 'osmd.cursor.hide();');
                            },
                          ),
                        ),
                        Positioned(
                          top: 0,
                          left: MediaQuery.of(context).size.width *
                              playbackController.currentProgress,
                          child: Container(
                            width: 30,
                            height: imageHeight,
                            color: const Color(0xffeb8e8e).withOpacity(0.4),
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 12),

                    // 다음 악보 흐릿하게
                    Opacity(
                      opacity: 0.3,
                      child: SizedBox(
                        height: imageHeight,
                        child: InAppWebView(
                          initialFile: 'assets/web/index.html',
                          onWebViewCreated: (controller) async {
                            controller.addJavaScriptHandler(
                              handlerName: 'sendFileToOSMD',
                              callback: (args) async {
                                String xml = await rootBundle
                                    .loadString('assets/music/demo.xml');
                                return xml;
                              },
                            );
                            await controller.evaluateJavascript(
                              source: 'loadLine($nextPage);',
                            );
                          },
                        ),
                      ),
                    ),
                  ],
                ),

                const Spacer(),

                // 📊 진행 바
                Column(
                  children: [
                    Container(
                      height: 7,
                      margin: const EdgeInsets.symmetric(horizontal: 120),
                      decoration: BoxDecoration(
                        color: Colors.white,
                        boxShadow: [
                          BoxShadow(
                            color: const Color(0xffd9d9d9),
                            blurRadius: 4,
                            offset: const Offset(0, 4),
                          ),
                        ],
                        borderRadius: BorderRadius.circular(20),
                      ),
                      width: double.infinity,
                      alignment: Alignment.centerLeft,
                      child: FractionallySizedBox(
                        widthFactor:
                            playbackController.currentDuration.inMilliseconds /
                                playbackController.totalDuration.inMilliseconds,
                        child: ClipRRect(
                          borderRadius: BorderRadius.circular(20),
                          child: Container(
                            height: 7,
                            decoration:
                                const BoxDecoration(color: Color(0xffEB8E8E)),
                          ),
                        ),
                      ),
                    ),
                    const SizedBox(height: 10),
                    Padding(
                      padding: const EdgeInsets.symmetric(horizontal: 122),
                      child: Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          Text(
                            '${playbackController.currentDuration.inMinutes}:${(playbackController.currentDuration.inSeconds % 60).toString().padLeft(2, '0')}',
                            style: const TextStyle(fontSize: 12),
                          ),
                          const Text('1:00', style: TextStyle(fontSize: 12)),
                        ],
                      ),
                    )
                  ],
                ),
              ],
            ),
          ),
          if (playbackController.isCountingDown)
            Container(
              color: Colors.black.withOpacity(0.6),
              child: Center(
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                  children: List.generate(3, (i) {
                    int number = 3 - i;
                    return Padding(
                      padding: const EdgeInsets.symmetric(horizontal: 32),
                      child: Stack(
                        alignment: Alignment.center,
                        children: [
                          Text(
                            '$number',
                            style: TextStyle(
                              fontSize: 72,
                              fontWeight: FontWeight.bold,
                              foreground: Paint()
                                ..style = PaintingStyle.stroke
                                ..strokeWidth = 10
                                ..color = playbackController.countdown == number
                                    ? const Color(0xffB95D4C)
                                    : const Color(0xff949494),
                            ),
                          ),
                          Text(
                            '$number',
                            style: TextStyle(
                              fontSize: 72,
                              fontWeight: FontWeight.bold,
                              color: playbackController.countdown == number
                                  ? const Color(0xffFD9B8A)
                                  : const Color(0xfff6f6f6),
                            ),
                          ),
                        ],
                      ),
                    );
                  }),
                ),
              ),
            ),
        ],
      ),
    );
  }
}
