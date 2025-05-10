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
import 'dart:io';
import 'package:flutter_sound/flutter_sound.dart' as fs;
import 'package:flutter_sound/public/flutter_sound_recorder.dart';
import 'package:stomp_dart_client/stomp_dart_client.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import 'package:xml/xml.dart';
import 'package:path_provider/path_provider.dart';
import '../../widgets/drum_recording_widget.dart';

class DrumSheetPlayer extends StatefulWidget {
  const DrumSheetPlayer({super.key});

  @override
  State<DrumSheetPlayer> createState() => _DrumSheetPlayerState();
}

class _DrumSheetPlayerState extends State<DrumSheetPlayer> {
  late PlaybackController playbackController;
  late OSMDService osmdService;
  bool _isControllerInitialized = false;
  // 녹음/웹소켓 관련 변수
  late fs.FlutterSoundRecorder _recorder;
  late StompClient _stompClient;
  final _storage = const FlutterSecureStorage();
  Timer? _recordingDataTimer;
  String? _recordingPath;
  bool _isRecording = false;
  bool _webSocketConnected = false;
  String _userEmail = '';
  int _beatsPerMeasure = 4;
  int _totalMeasures = 1;
  double _bpm = 60.0;
  int _currentMeasure = 0;

  // DrumRecordingWidget 관련 변수
  final GlobalKey<DrumRecordingWidgetState> _drumRecordingKey = GlobalKey();
  List<dynamic> _detectedOnsets = [];
  final String _recordingStatusMessage = '';

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    if (!_isControllerInitialized) {
      final imageHeight = MediaQuery.of(context).size.height * 0.27;
      playbackController = PlaybackController(imageHeight: imageHeight)
        ..onProgressUpdate = (progress) {
          setState(() {});
        }
        ..onPlaybackStateChange = (isPlaying) {
          setState(() {});
        }
        ..onCountdownUpdate = (count) {
          setState(() {});
        }
        ..onPageChange = (page) async {
          setState(() {});
        };
      _isControllerInitialized = true;
      // osmdService 초기화도 여기에 넣어도 됩니다.
    }
  }

  @override
  void initState() {
    super.initState();
    _initRecorder();
    _setupWebSocket();

    // OSMDService 초기화할 때 onDataLoaded 연결
    osmdService = OSMDService(
      onDataLoaded: ({
        required Uint8List base64Image,
        required Map<String, dynamic> json,
        required double bpm,
        required double canvasWidth,
        required double canvasHeight,
        required List<dynamic> lineBounds,
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

  Future<void> _initRecorder() async {
    var status = await Permission.microphone.request();
    if (status != PermissionStatus.granted) {
      throw fs.RecordingPermissionException('마이크 권한이 부여되지 않았습니다.');
    }
    _recorder = fs.FlutterSoundRecorder();
    await _recorder.openRecorder();
    final appDocDir = await getApplicationDocumentsDirectory();
    _recordingPath = '${appDocDir.path}/drum_performance.wav';
  }

  Future<void> _setupWebSocket() async {
    final token = await _storage.read(key: 'access_token');
    _userEmail = await _storage.read(key: 'user_email') ?? 'test@example.com';
    _stompClient = StompClient(
      config: StompConfig.sockJS(
        url: 'http://34.68.164.98:28080/ws/audio',
        onConnect: (StompFrame frame) {
          print('✅ WebSocket 연결 완료!');
          _webSocketConnected = true;
        },
        beforeConnect: () async => print('🌐 WebSocket 연결 시도 중...'),
        onWebSocketError: (dynamic error) {
          print('❌ WebSocket 오류 발생: $error');
        },
        onDisconnect: (frame) {
          print('🔌 WebSocket 연결 끊어짐');
          _webSocketConnected = false;
        },
        stompConnectHeaders: {
          'Authorization': token ?? '',
        },
      ),
    );
    _stompClient.activate();
  }

  Future<void> _startRecording() async {
    if (_isRecording) return;
    if (!_webSocketConnected) {
      print('❌ 녹음을 시작할 수 없습니다: WebSocket이 연결되지 않았습니다.');
      return;
    }
    await _recorder.startRecorder(
      toFile: _recordingPath,
      codec: fs.Codec.pcm16WAV,
      sampleRate: 16000,
      numChannels: 1,
      bitRate: 16000,
    );
    _isRecording = true;
    _currentMeasure = 0;
    final measureSeconds = (_beatsPerMeasure * 60.0) / _bpm;
    _recordingDataTimer =
        Timer.periodic(Duration(seconds: measureSeconds.toInt()), (timer) {
      _sendRecordingDataWithMeasure();
    });
  }

  Future<void> _stopRecording() async {
    if (!_isRecording) return;
    _recordingDataTimer?.cancel();
    await _recorder.stopRecorder();
    _isRecording = false;
    print('🎙️ 녹음 종료');
  }

  Future<void> _sendRecordingDataWithMeasure() async {
    if (!_stompClient.connected) {
      print('❌ WebSocket 연결이 되지 않아 데이터 전송 실패');
      return;
    }
    try {
      final file = File(_recordingPath!);
      if (await file.exists()) {
        final base64String = base64Encode(await file.readAsBytes());
        final message = {
          'email': _userEmail,
          'message': base64String,
          'currentMeasure': _currentMeasure,
          'totalMeasures': _totalMeasures
        };
        print(
            '📤 녹음 데이터 전송: ${DateTime.now()} (마디: ${_currentMeasure + 1}/$_totalMeasures)');
        _stompClient.send(
          destination: '/app/audio/forwarding',
          body: json.encode(message),
          headers: {'content-type': 'application/json'},
        );
        _currentMeasure++;
        if (_currentMeasure >= _totalMeasures) {
          _stopRecording();
        }
      }
    } catch (e) {
      print('❌ 녹음 데이터 전송 중 오류 발생: $e');
    }
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
                                    playbackController.stopPlayback();
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
                                          playbackController.stopPlayback();
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
                                                // 이미 멈춰있으니 추가 동작 불필요
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
                                              // 재생 중일 때는 배속 변경 못하도록 함
                                              if (!playbackController
                                                  .isPlaying) {
                                                playbackController.setSpeed(s);
                                              }
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
                          onTap: () {
                            if (playbackController.isPlaying) {
                              playbackController.stopPlayback();
                              // 녹음 중지
                              _drumRecordingKey.currentState?.stopRecording();
                            } else {
                              // 녹음 시작
                              _drumRecordingKey.currentState?.startCountdown(
                                onCountdownComplete: () {
                                  _drumRecordingKey.currentState
                                      ?.startRecording();
                                  playbackController.showCountdownAndStart();
                                },
                              );
                            }
                          },
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
                Column(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    // 현재 줄 악보
                    Container(
                      height: imageHeight,
                      margin:
                          const EdgeInsets.only(bottom: 12), // 현재 줄과 다음 줄 간격
                      decoration: BoxDecoration(
                        color: Colors.white,
                        borderRadius: BorderRadius.circular(5),
                        boxShadow: [
                          BoxShadow(
                            color: Colors.black.withOpacity(0.08),
                            blurRadius: 6,
                            offset: Offset(0, 4),
                          ),
                        ],
                      ),
                      child: ClipRRect(
                        borderRadius: BorderRadius.circular(5),
                        child: LayoutBuilder(
                          builder: (context, constraints) {
                            // 실제 악보가 그려지는 폭
                            final displayWidth = constraints.maxWidth;
                            return Stack(
                              children: [
                                // 재생했거나 재생 중이거나 재생 끝난 뒤에도(=paused 상태 포함) 커서 계속 표시
                                if (playbackController.currentDuration >
                                        Duration.zero ||
                                    playbackController.isPlaying ||
                                    playbackController.currentDuration >=
                                        playbackController.totalDuration)
                                  CursorWidget(
                                    cursor: playbackController.currentCursor,
                                    imageWidth: displayWidth,
                                    height: imageHeight,
                                  ),
                                if (playbackController.currentLineImage != null)
                                  Image.memory(
                                    playbackController.currentLineImage!,
                                    width: displayWidth,
                                    height: imageHeight,
                                    fit: BoxFit.fitWidth,
                                  ),
                              ],
                            );
                          },
                        ),
                      ),
                    ),

                    // 👀 다음 줄 미리보기
                    if (playbackController.nextLineImage != null)
                      Container(
                        height: imageHeight,
                        margin: const EdgeInsets.only(bottom: 5),
                        decoration: BoxDecoration(
                          // 흰색의 100% → 예: 80% 불투명(20% 투명)으로 조절
                          color: Colors.white.withOpacity(0.8),
                          borderRadius: BorderRadius.circular(5),
                          boxShadow: [
                            BoxShadow(
                              color: Colors.black.withOpacity(0.08),
                              blurRadius: 6,
                              offset: Offset(0, 4),
                            ),
                          ],
                        ),
                        child: ClipRRect(
                          borderRadius: BorderRadius.circular(5),
                          child: Opacity(
                            // 악보만 50% 투명
                            opacity: 0.5,
                            child: Image.memory(
                              playbackController.nextLineImage!,
                              width: double.infinity,
                              height: imageHeight,
                              fit: BoxFit.fitWidth,
                            ),
                          ),
                        ),
                      ),
                  ],
                ),

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
                                  height: 7, color: const Color(0xffD97D6C)),
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

                // 녹음 상태 메시지 표시
                if (_recordingStatusMessage.isNotEmpty)
                  Padding(
                    padding: const EdgeInsets.only(top: 8),
                    child: Text(
                      _recordingStatusMessage,
                      style: const TextStyle(
                        color: Color(0xFFE5958B),
                        fontSize: 14,
                        fontWeight: FontWeight.bold,
                      ),
                      textAlign: TextAlign.center,
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

          // DrumRecordingWidget 추가 (보이지 않지만 기능 사용)
          Offstage(
            offstage: true,
            child: DrumRecordingWidget(
              key: _drumRecordingKey,
              title: playbackController.sheetInfo?.title ?? '',
              xmlFilePath: 'assets/music/demo.xml',
              audioFilePath: 'assets/music/demo.wav',
              onRecordingComplete: (onsets) {
                setState(() {
                  _detectedOnsets = onsets;
                });
              },
              onOnsetsReceived: (onsets) {
                setState(() {
                  _detectedOnsets = onsets;
                });
              },
              onMusicXMLParsed: (info) {
                setState(() {
                  _beatsPerMeasure = info['beatsPerMeasure'] as int;
                  _totalMeasures = info['totalMeasures'] as int;
                  _bpm = info['bpm'] as double;
                });
              },
            ),
          ),

          // DrumRecordingWidget의 카운트다운 오버레이 표시
          if (_drumRecordingKey.currentState != null)
            _drumRecordingKey.currentState!.buildCountdownOverlay(),
        ],
      ),
    );
  }
}
