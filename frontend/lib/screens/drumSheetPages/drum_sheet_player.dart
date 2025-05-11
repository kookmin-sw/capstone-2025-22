import 'dart:io';
import 'dart:async';
import 'dart:convert';
import 'package:logger/logger.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';
import 'package:flutter_sound/flutter_sound.dart' as fs;
import 'package:stomp_dart_client/stomp_dart_client.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';

import '../../models/cursor.dart';
import '../../models/sheet_info.dart';
import '../../services/osmd_service.dart';
import '../../services/scoring_service.dart';
import '../../services/api_func.dart';
import '../../widgets/drum_recording_widget.dart';
import './widgets/cursor_widget.dart';
import './widgets/confirmation_dialog.dart';
import 'playback_controller.dart';

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
  int _currentMeasure = 0; // 녹음 마디 단위로 관리, 0-based

  // DrumRecordingWidget 관련 변수
  final GlobalKey<DrumRecordingWidgetState> _drumRecordingKey = GlobalKey();
  List<dynamic> _detectedOnsets = [];
  final String _recordingStatusMessage = '';

  // 채점 기능 관련 변수
  late ScoringService scoringService;
  String practiceIdentifier = '';
  int userSheetId = 0; // 현재 악보의 실제 ID (반드시 설정)
  List<Cursor> missedNotes = []; // 틀린 음표 저장
  int _currentMeasureOneBased = 0; // 직전 마디 채점 트리거, 1-based

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
        ..onPlaybackComplete = (lastMeasure) {
          _simulateDummyScoring(lastMeasure);
        } // 테스트 용 (제거하기)
        ..onCountdownUpdate = (count) {
          setState(() {});
        }
        ..onPageChange = (page) async {
          setState(() {});
        }
        // 마디 번호가 바뀌었을 때만 (테스트 용)
        ..onCursorMove = (cursor) {
          if (!playbackController.isPlaying) return;
          // OSMD가 주는 0-based → 화면/채점용 1-based 로 변환
          final newMeasure = cursor.measureNumber + 1;

          // measureNumber가 바뀔 때만
          if (newMeasure != _currentMeasureOneBased) {
            setState(() {
              // 바로 전에 연주를 마친 마디 번호가 1 이상이면 스코어링
              if (_currentMeasureOneBased >= 1) {
                _simulateDummyScoring(_currentMeasureOneBased);
              }
              // 그 다음 현재 마디 번호 갱신
              _currentMeasureOneBased = newMeasure;
            });
          }
        };
      _isControllerInitialized = true;
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
          // 줄별 이미지 메모리 캐시
          for (final bytes in lineImages) {
            final provider = MemoryImage(bytes);
            await precacheImage(provider, context);
          }

          final rawCursorList = (json['rawCursorList'] as List)
              .map((e) => Cursor.fromJson(e))
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
            userSheetId = int.tryParse(sheetInfo.id) ?? 0; // 악보 ID int로 파싱
            playbackController.canvasWidth = canvasWidth;
            playbackController.rawCursorList = rawCursorList; // 1차 채점용으로 추가
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

  // 최초 identifier 요청 및 웹소켓 결과 구독
  Future<void> _startPracticeSession() async {
    final token = await _storage.read(key: 'access_token');

    final response = await postHTTP(
      '/audio/practice',
      null, // 요청 본문이 없으므로 null 전달
      reqHeader: {
        'authorization': token ?? '',
      },
    );
    if (response['errMessage'] == null) {
      setState(() {
        practiceIdentifier = response['body'];
      });
      scoringService.subscribeToScoringResults(
          _userEmail, _handleScoringResult);
    } else {
      print("Error: ${response['errMessage']}");
      // 추가적인 오류 처리 (UI에 에러 표시 등)
    }
  }

  Future<String?> _fetchPracticeIdentifier() async {
    final token = await _storage.read(key: 'access_token');
    final response = await postHTTP('/audio/practice', null,
        reqHeader: {'authorization': token ?? ''});
    if (response['errMessage'] == null) {
      return response['body'] as String;
    } else {
      print('Identifier 요청 실패: ${response['errMessage']}');
      return null;
    }
  }

  // 실시간 채점 결과 처리 로직
  void _handleScoringResult(Map<String, dynamic> scoringResult) {
    print('📥 채점 결과: $scoringResult');

    final measureNumber = scoringResult['measureNumber'];
    final answerOnsetPlayed = scoringResult['answerOnsetPlayed'];
    final measureIndex = int.parse(measureNumber) - 1;

    // 현재 줄의 measureNumber 범위 확인
    final measuresPerLine = 4;
    final currentLineStart = playbackController.currentPage * measuresPerLine;
    final currentLineEnd = currentLineStart + measuresPerLine;

    if (measureIndex >= currentLineStart && measureIndex < currentLineEnd) {
      // 틀린 박자 위치 (answerOnsetPlayed가 false인 index만)
      final missedNotesIndices = <int>[];
      for (int i = 0; i < answerOnsetPlayed.length; i++) {
        if (!answerOnsetPlayed[i]) {
          missedNotesIndices.add(i);
        }
      }

      // missedNotesIndices를 기반으로 커서 회색 표시 로직 추가
      playbackController.addMissedNotesCursor(
        measureIndex: measureIndex,
        missedIndices: missedNotesIndices,
      );
      setState(() {}); // UI 갱신
    }
  }

// 커서 인덱스 계산 (helper method 추가)
  int cursorListIndex(Cursor cursor) {
    final cursorsInMeasure = playbackController.sheetInfo!.cursorList
        .where((c) => c.measureNumber == cursor.measureNumber)
        .toList();
    // ts가 같은 애를 기준으로 찾자
    final idx = cursorsInMeasure.indexWhere((c) => c.ts == cursor.ts);
    return idx;
  }

  // 테스트 용
  void _simulateDummyScoring(int measureNumber) {
    final dummy = {
      'measureNumber': measureNumber.toString(),
      'userOnset': [0.1, 0.6, 1.2, 1.8],
      'answerOnset': [0.0, 0.5, 1.0, 1.5],
      'answerOnsetPlayed': [true, false, true, true],
      'matchedUserOnsetIndices': [0, -1, 2, 3],
    };
    _handleScoringResult(dummy);
  }

  @override
  void dispose() {
    playbackController.dispose();
    super.dispose();
  }

  // 1. _recordingPath를 하나의 고정된 경로로 설정하여 녹음 파일 덮어쓰기
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

  // 2. 녹음 시작 후, 마디별로 데이터를 웹소켓으로 전송 후 덮어쓰기
  Future<void> _startRecording() async {
    if (_isRecording) return;
    if (!_webSocketConnected) {
      print('❌ 녹음을 시작할 수 없습니다: WebSocket이 연결되지 않았습니다.');
      return;
    }
    await _recorder.startRecorder(
      toFile: _recordingPath, // 덮어쓰기 경로
      codec: fs.Codec.pcm16WAV,
      sampleRate: 16000,
      numChannels: 1,
      bitRate: 16000,
    );
    _isRecording = true;
    _currentMeasure = 0;

    // 배속을 감안한 한 마디의 길이 계산 (초 단위)
    final measureDuration =
        (_beatsPerMeasure * 60.0) / (_bpm * playbackController.speed);

    // Timer로 한 마디가 끝날 때마다 데이터를 전송
    _recordingDataTimer =
        Timer.periodic(Duration(seconds: measureDuration.toInt()), (timer) {
      _sendRecordingDataWithMeasure(); // 한 마디를 주기로 녹음 데이터 전송
    });
  }

  // 3. 녹음 중 데이터 전송 및 덮어쓰기
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
          'bpm': (_bpm * playbackController.speed).round(),
          'userSheetId': userSheetId, // TODO : 사용자 악보 ID 백엔드에서 받아와야 함
          'identifier': scoringService.identifier,
          'email': _userEmail,
          'message': base64String,
          'measureNumber': (_currentMeasure + 1).toString(),
          'endOfMeasure': (_currentMeasure + 1) >= _totalMeasures
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
          _stopRecording(); // 모든 마디 녹음 완료 후 종료
        }
      }
    } catch (e) {
      print('❌ 녹음 데이터 전송 중 오류 발생: $e');
    }
  }

  // 4. 녹음 종료
  Future<void> _stopRecording() async {
    if (!_isRecording) return;
    _recordingDataTimer?.cancel();
    await _recorder.stopRecorder();
    _isRecording = false;
    print('🎙️ 녹음 종료');
  }

  // 5. 녹음 일시 정지
  Future<void> _pauseRecording() async {
    if (!_isRecording) return;
    await _recorder.pauseRecorder(); // 녹음 일시정지
    _recordingDataTimer?.cancel(); // 전송 타이머도 멈춤
    print('⏸️ 녹음 일시정지');
  }

  //6. 녹음 재개
  Future<void> _resumeRecording() async {
    if (!_isRecording) return;
    await _recorder.resumeRecorder(); // 녹음 재개
    // measureDuration 는 기존 계산 그대로
    final measureDuration =
        (_beatsPerMeasure * 60.0) / (_bpm * playbackController.speed);
    _recordingDataTimer = Timer.periodic(
      Duration(seconds: measureDuration.toInt()),
      (_) => _sendRecordingDataWithMeasure(),
    );
    print('▶️ 녹음 재개 (마디 ${_currentMeasure + 1}부터)');
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

          scoringService = ScoringService(client: _stompClient);
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
                                                setState(() {
                                                  _currentMeasureOneBased =
                                                      0; // 1-based 마디 번호 초기화
                                                });
                                                _drumRecordingKey.currentState
                                                    ?.stopRecording();
                                                _stopRecording();
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
                          onTap: () async {
                            if (playbackController.isPlaying) {
                              // 재생 중이면 일시정지 & 녹음 중지
                              playbackController.stopPlayback();
                              _pauseRecording();
                            } else {
                              _drumRecordingKey.currentState?.startCountdown(
                                onCountdownComplete: () async {
                                  // ① 녹음 시작 전, identifier 받아오기
                                  final id = await _fetchPracticeIdentifier();
                                  if (id == null) {
                                    // 실패 시 사용자에게 알려주고 녹음 중단
                                    ScaffoldMessenger.of(context).showSnackBar(
                                        SnackBar(
                                            content:
                                                Text("채점 식별자를 가져오지 못했습니다.")));
                                    return;
                                  }
                                  scoringService.setIdentifier(id);
                                  scoringService.subscribeToScoringResults(
                                      _userEmail, _handleScoringResult);

                                  playbackController.showCountdownAndStart();
                                  if (_isRecording) {
                                    _resumeRecording();
                                  } else {
                                    _startRecording();
                                  }
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
                                for (final missed in playbackController
                                    .missedCursors
                                    .where((c) =>
                                        c.lineIndex ==
                                        playbackController.currentPage))
                                  CursorWidget(
                                    cursor: missed,
                                    imageWidth: displayWidth,
                                    height: imageHeight,
                                    decoration: BoxDecoration(
                                      color: const Color(0xFFE1E1E1),
                                      borderRadius: BorderRadius.circular(4),
                                    ),
                                  ),
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
                                    gaplessPlayback: true,
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
                              gaplessPlayback: true,
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
              playbackController: playbackController, // playbackController 전달
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
