import 'dart:async';
import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import '../../models/sheet_info.dart';
import '../../models/cursor.dart';
import './widgets/cursor_widget.dart';
import 'playback_controller.dart';
import './widgets/confirmation_dialog.dart';
import '../../services/osmd_service.dart';

class DrumSheetPlayer extends StatefulWidget {
  const DrumSheetPlayer({super.key});

  @override
  State<DrumSheetPlayer> createState() => _DrumSheetPlayerState();
}

class _DrumSheetPlayerState extends State<DrumSheetPlayer> {
  late PlaybackController playbackController;
  late OSMDService osmdService;

  @override
  void initState() {
    super.initState();

    playbackController = PlaybackController()
      ..onProgressUpdate = (progress) {
        setState(() {
          // 진행 바 위치 업데이트
        });
      }
      ..onPlaybackStateChange = (isPlaying) {
        setState(() {
          // 재생 / 일시정지 아이콘 상태 변경
        });
      }
      ..onCountdownUpdate = (count) {
        setState(() {
          // 카운트다운 숫자 표시
        });
      }
      ..onPageChange = (page) async {
        setState(() {
          // 현재 재생 중인 줄 (page) 업데이트
        });
      };

    // OSMDService 초기화할 때 onDataLoaded 연결
    osmdService = OSMDService(
      onDataLoaded: ({
        required Uint8List base64Image,
        required Map<String, dynamic> json,
        required double bpm,
        required double canvasWidth,
        required double canvasHeight,
      }) async {
        try {
          final int totalLines = (json['lineCount'] is int)
              ? json['lineCount'] as int
              : (json['lineCount'] ?? 1).toInt();
          final List<Uint8List> lineImages =
              (json['lineImages'] as List<dynamic>)
                  .map((e) => base64Decode(e))
                  .toList();

          final sheetInfo = SheetInfo(
            id: '', // 일단 빈 값 (추후 백엔드 연동시 수정)
            title: '그라데이션',
            artist: '10CM',
            bpm: bpm.toInt(),
            canvasHeight: canvasHeight,
            cursorList: (json['cursorList'] as List<dynamic>)
                .map((e) => Cursor.fromJson(e))
                .toList(),
            fullSheetImage: base64Image,
            xmlData: json['xmlData'] as String?,
            lineImages: lineImages,
            createdDate: DateTime.now(),
          );

          setState(() {
            playbackController.loadSheetInfo(sheetInfo);
            playbackController.canvasWidth = canvasWidth;
            playbackController
                .calculateTotalDurationFromCursorList(bpm); // 총 재생시간 계산

            playbackController.currentLineImage =
                lineImages.isNotEmpty ? lineImages[0] : null;
            playbackController.nextLineImage =
                lineImages.length > 1 ? lineImages[1] : null;
          });
        } catch (e, st) {
          debugPrint('🔴 onDataLoaded error: $e\n$st');
        }
      },
    );
    Future.microtask(() async {
      final xmlData = await rootBundle.load('assets/music/demo.xml');
      if (!mounted) return;
      await osmdService.startOSMDService(
        xmlData: xmlData.buffer.asUint8List(),
        pageWidth: 1080,
      );
    });
  }

  @override
  void dispose() {
    playbackController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final imageHeight =
        MediaQuery.of(context).size.height * 0.27; // 악보 이미지 영역 높이
    if (playbackController.sheetInfo == null) {
      return const Center(child: CircularProgressIndicator());
    }

    return Scaffold(
      backgroundColor: const Color(0xFFF5F5F5),
      body: Stack(
        children: [
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 40),
            child: Column(
              children: [
                // 🎵 상단 컨트롤 바 (홈버튼, 제목, 재생, 속도)
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
                                // 홈 버튼 눌렀을 때
                                GestureDetector(
                                  onTap: () {
                                    showDialog(
                                      context: context,
                                      barrierDismissible: true,
                                      builder: (_) => ConfirmationDialog(
                                        message: "메인으로 이동하시겠습니까?",
                                        onConfirm: () {
                                          Navigator.of(context).pop();
                                          // TODO: 메인 이동 로직
                                        },
                                        onCancel: () {
                                          Navigator.of(context).pop();
                                          playbackController
                                              .stopPlayback(); // 취소하면 정지 상태
                                        },
                                      ),
                                    );
                                  },
                                  child: const Icon(Icons.home,
                                      size: 30, color: Color(0xff646464)),
                                ),

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
                                      '${playbackController.sheetInfo!.title} - ${playbackController.sheetInfo!.artist}',
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
                                      child: // 리셋 버튼 눌렀을 때
                                          GestureDetector(
                                        onTap: () {
                                          showDialog(
                                            context: context,
                                            barrierDismissible: true,
                                            builder: (_) => ConfirmationDialog(
                                              message: "처음부터 다시 연주하시겠습니까?",
                                              onConfirm: () {
                                                Navigator.of(context).pop();
                                                playbackController
                                                    .resetToStart(); // 리셋 로직 실행
                                              },
                                              onCancel: () {
                                                Navigator.of(context).pop();
                                                playbackController
                                                    .stopPlayback(); // 취소하면 정지 상태
                                              },
                                            ),
                                          );
                                        },
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
                // Column(
                //   crossAxisAlignment: CrossAxisAlignment.stretch,
                //   children: [
                //     // 현재 줄 악보
                //     Container(
                //       height: imageHeight,
                //       margin:
                //           const EdgeInsets.only(bottom: 12), // 현재 줄과 다음 줄 간격
                //       decoration: BoxDecoration(
                //         color: Colors.white,
                //         borderRadius: BorderRadius.circular(5),
                //         boxShadow: [
                //           BoxShadow(
                //             color: Colors.black.withOpacity(0.08),
                //             blurRadius: 6,
                //             offset: Offset(0, 4),
                //           ),
                //         ],
                //       ),
                //       child: ClipRRect(
                //         borderRadius: BorderRadius.circular(5),
                //         child: Stack(
                //           children: [
                //             CursorWidget(
                //               cursor: playbackController.currentCursor,
                //               imageWidth: MediaQuery.of(context).size.width,
                //               canvasWidth: playbackController.canvasWidth,
                //             ),
                //             if (playbackController.currentLineImage != null)
                //               Image.memory(
                //                 playbackController.currentLineImage!,
                //                 width: double.infinity,
                //                 height: imageHeight,
                //                 fit: BoxFit.fitWidth,
                //               ),
                //           ],
                //         ),
                //       ),
                //     ),

                //     // 👀 다음 줄 미리보기
                //     if (playbackController.nextLineImage != null)
                //       Container(
                //         height: imageHeight,
                //         margin: const EdgeInsets.only(bottom: 5),
                //         decoration: BoxDecoration(
                //           // 흰색의 100% → 예: 80% 불투명(20% 투명)으로 조절
                //           color: Colors.white.withOpacity(0.8),
                //           borderRadius: BorderRadius.circular(5),
                //           boxShadow: [
                //             BoxShadow(
                //               color: Colors.black.withOpacity(0.08),
                //               blurRadius: 6,
                //               offset: Offset(0, 4),
                //             ),
                //           ],
                //         ),
                //         child: ClipRRect(
                //           borderRadius: BorderRadius.circular(5),
                //           child: Opacity(
                //             // 악보만 50% 투명
                //             opacity: 0.5,
                //             child: Image.memory(
                //               playbackController.nextLineImage!,
                //               width: double.infinity,
                //               height: imageHeight,
                //               fit: BoxFit.fitWidth,
                //             ),
                //           ),
                //         ),
                //       ),
                //   ],
                // ),

                Spacer(flex: 2),

                // 📊 진행 바 + 시간 Row
                Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 120), // 좌우 마진
                  child: Row(
                    children: [
                      // 현재 재생 시간
                      Text(
                        '${playbackController.currentDuration.inMinutes}:'
                        '${(playbackController.currentDuration.inSeconds % 60).toString().padLeft(2, '0')}',
                        style: const TextStyle(fontSize: 13),
                      ),

                      const SizedBox(width: 18), // 시간과 바 사이 간격

                      // 진행 바
                      Expanded(
                        child: Container(
                          height: 7,
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
                          child: FractionallySizedBox(
                            alignment: Alignment.centerLeft,
                            widthFactor: (playbackController
                                        .totalDuration.inMilliseconds ==
                                    0)
                                ? 0.0
                                : (playbackController
                                            .currentDuration.inMilliseconds /
                                        playbackController
                                            .totalDuration.inMilliseconds)
                                    .clamp(0.0, 1.0),
                            child: ClipRRect(
                              borderRadius: BorderRadius.circular(20),
                              child: Container(
                                  height: 7, color: const Color(0xffEB8E8E)),
                            ),
                          ),
                        ),
                      ),

                      const SizedBox(width: 18), // 바와 전체 시간 사이 간격

                      // 전체 재생 시간
                      Text(
                        '${playbackController.totalDuration.inMinutes}:'
                        '${(playbackController.totalDuration.inSeconds % 60).toString().padLeft(2, '0')}',
                        style: const TextStyle(fontSize: 13),
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ),

          // ⏱️ 카운트다운 오버레이
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
