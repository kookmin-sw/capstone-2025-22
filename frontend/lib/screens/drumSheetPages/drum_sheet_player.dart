// ignore_for_file: avoid_print

import 'dart:io';
import 'dart:async';
import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';
import 'package:flutter_sound/flutter_sound.dart' as fs;
import 'package:stomp_dart_client/stomp_dart_client.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';

import '../../models/cursor.dart';
import '../../models/sheet_info.dart';
import '../../services/osmd_service.dart';
import '../../services/scoring_service.dart';
import '../../services/api_func.dart';
import '../../widgets/drum_recording_widget.dart';
import './widgets/cursor_widget.dart';
import './widgets/confirmation_dialog.dart';
import './practice_result_MS.dart';
import 'playback_controller.dart';
import 'package:capstone_2025/screens/mainPages/navigation_screens.dart';
import 'package:capstone_2025/screens/drumSheetPages/sheetXmlDataTemp.dart'
    as sheet_xml_data_temp;

class DrumSheetPlayer extends StatefulWidget {
  final int sheetId;
  final String title;
  final String artist;
  final String sheetXmlData;

  const DrumSheetPlayer({
    super.key,
    this.sheetId = 7, // 테스트 값 넣어둠
    this.title = 'FOREVER', // '그라데이션'
    this.artist = 'BABY MONSTER', // '10CM'
    // 기존의 demo.xml을 base64로 변환한 것
    this.sheetXmlData = sheet_xml_data_temp.temp,
  });

  @override
  State<DrumSheetPlayer> createState() => _DrumSheetPlayerState();
}

class _DrumSheetPlayerState extends State<DrumSheetPlayer> {
  // ===== 컨트롤러 및 서비스 =====
  late PlaybackController playbackController;
  late OSMDService osmdService;
  late ScoringService scoringService;
  bool _isControllerInitialized = false;

  // ===== 악보 및 XML 관련 변수 =====
  late String xmlDataString;
  int _beatsPerMeasure = 4;
  int _totalMeasures = 1;
  double _bpm = 60.0;

  // ===== 재생 및 마디 관련 변수 =====
  final int _currentMeasure = 0; // 녹음 마디 (0-based)
  int _currentMeasureOneBased = 0; // 채점용 마디 (1-based)

  // ===== 녹음 및 웹소켓 관련 변수 =====
  late fs.FlutterSoundRecorder _recorder;
  late StompClient _stompClient;
  final _storage = const FlutterSecureStorage();
  Timer? _recordingDataTimer;
  String? _recordingPath;
  final bool _isRecording = false;
  final bool _webSocketConnected = false;
  final String _userEmail = '';

  // ===== 채점 및 결과 관련 변수 =====
  final GlobalKey<DrumRecordingWidgetState> _drumRecordingKey = GlobalKey();
  List<dynamic> _detectedOnsets = [];
  String practiceIdentifier = '';
  int userSheetId = 0;
  List<Cursor> missedNotes = [];
  final List<Map<String, dynamic>> _beatGradingResults = [];

  @override
  void initState() {
    super.initState();

    // XML 데이터 로드
    Future.microtask(() async {
      await _loadXMLDataFromBackend();
    });

    // OSMD 서비스 초기화 - 악보 렌더링 처리
    _initializeOSMDService();
  }

  /// OSMD 서비스 초기화 메서드
  void _initializeOSMDService() {
    osmdService = OSMDService(
      onDataLoaded: _handleOSMDDataLoaded,
    );
  }

  /// OSMD 데이터 로드 완료 시 처리 콜백
  Future<void> _handleOSMDDataLoaded({
    required Uint8List base64Image,
    required Map<String, dynamic> json,
    required double bpm,
    required double canvasWidth,
    required double canvasHeight,
    required List<dynamic> lineBounds,
    required int totalMeasures,
  }) async {
    // 악보 상세 페이지에서 악보 전체 이미지 사용하기 위해 로컬에 저장
    try {
      final dir = await getApplicationDocumentsDirectory();
      final previewPath = '${dir.path}/sheet_preview_${widget.sheetId}.png';
      final file = File(previewPath);

      if (!await file.exists()) {
        // 파일이 없을 때만 생성
        await file.writeAsBytes(base64Image, flush: true);
        debugPrint('📁 preview 이미지 저장: $previewPath');
      } else {
        debugPrint('📁 preview 이미지 이미 존재, 스킵');
      }
    } catch (e) {
      debugPrint('⚠️ Preview save failed: $e');
    }

    try {
      // 1. 기본 데이터 추출
      final int totalLines = (json['lineCount'] is int)
          ? json['lineCount'] as int
          : (json['lineCount'] ?? 1).toInt();

      // 2. 줄별 이미지 데이터 처리
      final List<Uint8List> lineImages = (json['lineImages'] as List<dynamic>)
          .map((e) => base64Decode(e))
          .toList();

      // 3. 이미지 캐싱 처리
      for (final bytes in lineImages) {
        final provider = MemoryImage(bytes);
        await precacheImage(provider, context);
      }

      // 4. 커서 정보 추출
      final rawCursorList = (json['rawCursorList'] as List)
          .map((e) => Cursor.fromJson(e))
          .toList();

      // 5. 악보 정보 객체 생성
      final sheetInfo = SheetInfo(
        id: widget.sheetId.toString(),
        title: widget.title,
        artist: widget.artist,
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

      // 6. 상태 업데이트
      setState(() {
        // 컨트롤러에 악보 정보 로드
        playbackController.loadSheetInfo(sheetInfo);
        userSheetId = int.tryParse(sheetInfo.id) ?? 0;
        playbackController.canvasWidth = canvasWidth;
        playbackController.rawCursorList = rawCursorList;
        playbackController.calculateTotalDurationFromCursorList(bpm);
        playbackController.totalMeasures = totalMeasures;

        // 현재/다음 줄 이미지 설정
        playbackController.currentLineImage =
            lineImages.isNotEmpty ? lineImages[0] : null;
        playbackController.nextLineImage =
            lineImages.length > 1 ? lineImages[1] : null;
      });
    } catch (e, st) {
      debugPrint('🔴 OSMD 데이터 로드 에러: $e\n$st');
    }
  }

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    if (!_isControllerInitialized) {
      _initializePlaybackController();
      _isControllerInitialized = true;
    }
  }

  void _initializePlaybackController() {
    final imageHeight = MediaQuery.of(context).size.height * 0.27;
    playbackController = PlaybackController(imageHeight: imageHeight)
      // 진행 업데이트 콜백
      ..onProgressUpdate = (progress) {
        setState(() {});
      }
      // 재생 상태 변경 콜백
      ..onPlaybackStateChange = (isPlaying) async {
        // setState(() {});
        // if (isPlaying) {
        //   // 연주 시작 시 identifier 요청
        //   final identifier = await fetchPracticeIdentifier();
        //   if (identifier != null) {
        //     if (_drumRecordingKey.currentState?.isRecording == true) {
        //       _drumRecordingKey.currentState?.resumeRecording();
        //     } else {
        //       _drumRecordingKey.currentState?.startRecording();
        //     }
        //   }
        // } else {
        //   _drumRecordingKey.currentState?.pauseRecording();
        // }
        setState(() {});
      }
      // 카운트다운 업데이트 콜백
      ..onCountdownUpdate = (count) {
        setState(() {});
      }
      // 페이지 변경 콜백
      ..onPageChange = (page) async {
        setState(() {});
      }
      // 커서 이동 콜백
      ..onCursorMove = (cursor) {
        if (!playbackController.isPlaying) return;
        // OSMD 0-based → 화면/채점용 1-based 변환
        final newMeasure = cursor.measureNumber + 1;

        // 마디 번호가 바뀔 때만 업데이트
        if (newMeasure != _currentMeasureOneBased) {
          setState(() {
            _currentMeasureOneBased = newMeasure;
          });
        }
      }
      // 마디 변경 시 녹음 처리 콜백 추가
      ..onMeasureChange = (measureNumber) {
        if (_drumRecordingKey.currentState != null) {
          // 현재 마디가 마지막 마디인지 확인
          final isLastMeasure = measureNumber == _totalMeasures - 1;
          // 마디별 녹음 데이터 전송
          _drumRecordingKey.currentState?.sendMeasureData(
            measureNumber: measureNumber + 1, // 1-based로 변환
            isLastMeasure: isLastMeasure,
          );
        }
      };
  }

  /// XML 데이터를 백엔드에서 로드
  Future<void> _loadXMLDataFromBackend() async {
    try {
      // 1. 백엔드에서 전달받은 base64 인코딩된 XML 데이터
      String base64XmlData = widget.sheetXmlData;

      // 2. base64를 바이트로 디코딩
      final Uint8List decodedXmlBytes = base64Decode(base64XmlData);

      // 3. UTF-8 문자열로 변환
      xmlDataString = utf8.decode(decodedXmlBytes);

      // 4. XML 선언 추가 (없는 경우)
      if (!xmlDataString.startsWith('<?xml')) {
        xmlDataString =
            '<?xml version="1.0" encoding="UTF-8"?>\n$xmlDataString';
      }

      // 5. OSMD 서비스 시작
      await osmdService.startOSMDService(
        xmlData: utf8.encode(xmlDataString),
        pageWidth: 1080,
      );
    } catch (e) {
      print("XML 데이터 로드 실패: $e");
    }
  }

  /// 연주 식별자 가져오기
  Future<String?> fetchPracticeIdentifier() async {
    try {
      final token = await _storage.read(key: 'access_token');
      if (token == null) {
        print('❌ 토큰이 없습니다');
        return null;
      }

      final response = await postHTTP('/audio/practice', null,
          reqHeader: {'authorization': token});

      if (response['body'] != null) {
        final identifier = response['body'] as String;
        print('✅ 연주 식별자 수신: $identifier');
        return identifier;
      } else {
        print('❌ 연주 식별자 요청 실패: ${response['message']}');
        return null;
      }
    } catch (e) {
      print('❌ 연주 식별자 요청 중 오류: $e');
      return null;
    }
  }

  /// 1차 채점 결과 처리
  void _handleScoringResult(Map<String, dynamic> scoringResult) {
    print('📥 채점 결과: $scoringResult');

    final measureNumber = scoringResult['measureNumber'];
    final answerOnsetPlayed = scoringResult['answerOnsetPlayed'];
    final measureIndex = int.parse(measureNumber) - 1;

    // 현재 줄의 마디 범위 확인
    final measuresPerLine = 4;
    final currentLineStart = playbackController.currentPage * measuresPerLine;
    final currentLineEnd = currentLineStart + measuresPerLine;

    // 현재 보이는 줄의 마디인 경우에만 처리
    if (measureIndex >= currentLineStart && measureIndex < currentLineEnd) {
      // 틀린 박자 위치 찾기 (false인 인덱스)
      final missedNotesIndices = <int>[];
      for (int i = 0; i < answerOnsetPlayed.length; i++) {
        if (!answerOnsetPlayed[i]) {
          missedNotesIndices.add(i);
        }
      }

      // 틀린 음표 커서 추가
      playbackController.addMissedNotesCursor(
        measureIndex: measureIndex,
        missedIndices: missedNotesIndices,
      );
      setState(() {}); // UI 갱신
    }
  }

  // 1차 채점 결과 메시지 처리
  void _onWsGradingMessage(Map<String, dynamic> msg) {
    print(
        "▶ 받은 채점 메시지 #${_beatGradingResults.length + 1}: measure=${msg['measureNumber']}, played=${msg['answerOnsetPlayed']}");
    setState(() {
      _beatGradingResults.add(msg);
      // 모든 마디 데이터가 수신되면 결과 처리
      if (_beatGradingResults.length == _totalMeasures) {
        print(
            "🗒️ 전체 마디 데이터 수신 완료: ${_beatGradingResults.map((m) => m['measureNumber']).toList()}");
        _applyGradingResults();
      }
    });
  }

  // 최종 채점 결과 적용 및 결과 화면 이동
  void _applyGradingResults() {
    print("✅ 1차 채점 완료: measureNumbers = "
        "${_beatGradingResults.map((m) => m['measureNumber']).toList()}");
    final initialBeatScore = computeScoreFrom1stGrading(_beatGradingResults);

    // 2초 딜레이 후 결과창으로 이동
    Future.delayed(const Duration(seconds: 2), () {
      Navigator.of(context).pushReplacement(
        MaterialPageRoute(
          builder: (_) => PracticeResultMS(
            sheetId: widget.sheetId,
            musicTitle: widget.title,
            musicArtist: widget.artist,
            score: initialBeatScore,
            xmlDataString: xmlDataString,
            practiceInfo: practiceInfo,
          ),
        ),
      );
    });
  }

  // 결과 저장을 위한 practiceInfo 변환
  List<Map<String, dynamic>> get practiceInfo {
    return _beatGradingResults.map((msg) {
      return {
        "measureNumber": msg["measureNumber"],
        "beatScoringResults": List<bool>.from(msg["answerOnsetPlayed"]),
        "finalScoringResults": <bool>[],
      };
    }).toList();
  }

  // 커서 인덱스 계산 (Helper 메서드)
  int cursorListIndex(Cursor cursor) {
    final cursorsInMeasure = playbackController.sheetInfo!.cursorList
        .where((c) => c.measureNumber == cursor.measureNumber)
        .toList();
    // 타임스탬프가 같은 커서 기준으로 찾기
    final idx = cursorsInMeasure.indexWhere((c) => c.ts == cursor.ts);
    return idx;
  }

  @override
  void dispose() {
    // 리소스 해제
    _recordingDataTimer?.cancel();
    super.dispose();
  }

  // 1차 채점 데이터로 점수 계산
  int computeScoreFrom1stGrading(List<Map<String, dynamic>> results) {
    // 1) 각 마디별 결과를 하나의 리스트로 병합
    final allBeats =
        results.expand((m) => List<bool>.from(m['answerOnsetPlayed'])).toList();

    // 2) 틀린 음표 개수 세기
    final wrongCount = allBeats.where((b) => b == false).length;
    final totalCount = allBeats.length;

    if (totalCount == 0) return 0; // 예외 처리

    // 3) 100점 만점으로 환산
    final correctCount = totalCount - wrongCount;
    return ((correctCount / totalCount) * 100).round();
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
                                    // 오디오 재생 중지
                                    playbackController.stopPlayback();
                                    showDialog(
                                      context: context,
                                      barrierDismissible: true,
                                      builder: (_) => ConfirmationDialog(
                                        message: "메인으로 이동하시겠습니까?",
                                        onConfirm: () {
                                          print("다이얼로그 닫기 전");
                                          // 다이얼로그 닫기
                                          Navigator.of(context).pop();
                                          print("다이얼로그 닫음");

                                          // DrumRecordingWidget의 녹음 중지
                                          final drumRecordingState =
                                              _drumRecordingKey.currentState;
                                          if (drumRecordingState != null &&
                                              drumRecordingState.isRecording) {
                                            drumRecordingState.stopRecording();
                                          }
                                          print("녹음 중지");

                                          _beatGradingResults.clear();
                                          playbackController.missedCursors
                                              .clear();

                                          // 리소스 해제 - WebSocket 연결 종료
                                          _drumRecordingKey.currentState
                                              ?.cleanupResources();

                                          print("리소스 해제");

                                          // _recordingDataTimer 해제
                                          _recordingDataTimer?.cancel();
                                          print("타이머 해제");

                                          // 홈화면으로 이동: NavigationScreens 상태 업데이트 부분 수정
                                          WidgetsBinding.instance
                                              .addPostFrameCallback((_) {
                                            // 1. 먼저 현재 페이지를 스택에서 제거 (순서 변경)
                                            if (Navigator.canPop(context)) {
                                              Navigator.of(context).pop();
                                              print("현재 페이지 스택 제거 완료");
                                            }

                                            // 2. 그 다음 상위 위젯의 상태 업데이트
                                            final navigationScreensState =
                                                context.findAncestorStateOfType<
                                                    NavigationScreensState>();
                                            if (navigationScreensState !=
                                                    null &&
                                                navigationScreensState
                                                    .mounted) {
                                              navigationScreensState
                                                  .setState(() {
                                                navigationScreensState
                                                        .selectedIndex =
                                                    2; // 홈 화면 인덱스
                                              });
                                              print(
                                                  "NavigationScreens 상태 업데이트 완료");
                                            } else {
                                              print(
                                                  "NavigationScreensState를 찾을 수 없음");
                                              // 대안으로 직접 네비게이션 처리
                                              Navigator.of(context)
                                                  .pushAndRemoveUntil(
                                                MaterialPageRoute(
                                                  builder: (context) =>
                                                      const NavigationScreens(
                                                          firstSelectedIndex:
                                                              3),
                                                ),
                                                (route) =>
                                                    false, // 모든 이전 라우트 제거
                                              );
                                            }
                                          });
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
                                              onConfirm: () async {
                                                Navigator.of(context).pop();
                                                // 1) 녹음 중이면 중지하고 리소스 정리
                                                final recorder =
                                                    _drumRecordingKey
                                                        .currentState;
                                                if (recorder?.isRecording ==
                                                    true) {
                                                  await recorder!
                                                      .stopRecording();
                                                }
                                                // 2) 플레이어 & 내부 상태 리셋
                                                setState(() {
                                                  _currentMeasureOneBased = 0;
                                                  _beatGradingResults.clear();
                                                  playbackController
                                                      .missedCursors
                                                      .clear();
                                                  playbackController
                                                      .resetToStart();
                                                });
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
                              _drumRecordingKey.currentState?.pauseRecording();
                            } else {
                              setState(() {
                                _beatGradingResults.clear();
                                playbackController.missedCursors.clear();
                              });
                              playbackController.showCountdownAndStart();
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
              xmlDataString: xmlDataString,
              audioFilePath: '',
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
                print('info: $info');
                try {
                  // totalMeasures가 제대로 계산되었는지 확인
                  final totalMeasures = info['totalMeasures'] as int;
                  print('Total measures received: $totalMeasures');
                  // // XML 데이터를 파싱
                  // final document = XmlDocument.parse(
                  //     info['xmlData'] as String); // xmlData는 XML 문자열로 받아옴

                  // // 'measure' 태그를 찾아서 마디의 개수 구하기
                  // final measures = document.findAllElements('measure');
                  // final int totalMeasures =
                  //     measures.length; // measure의 개수를 totalMeasures로 설정
                  // print('Total measures: $totalMeasures'); // 마디의 개수 출력

                  // 기존 info에서 beatsPerMeasure, bpm 등 필요한 값을 가져오고, totalMeasures를 설정
                  setState(() {
                    _beatsPerMeasure = info['beatsPerMeasure'] as int;
                    _totalMeasures = totalMeasures; // 여기서 totalMeasures를 할당
                    _bpm = info['bpm'] as double;
                  });
                } catch (e) {
                  print('Error parsing XML: $e');
                }
              },
              onGradingResult: (msg) {
                _handleScoringResult(msg); // 1) 즉시 화면에 틀린 박자 커서 표시
                _onWsGradingMessage(msg); // 2) 리스트에 쌓아서, 마지막에 전체 점수 계산
              },
              playbackController: playbackController, //playbackController 전달
              fetchPracticeIdentifier: fetchPracticeIdentifier,
              userSheetId: widget.sheetId,
            ),
          ),
        ],
      ),
    );
  }
}
