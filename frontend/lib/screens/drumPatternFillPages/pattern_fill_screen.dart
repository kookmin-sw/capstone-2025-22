import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'package:capstone_2025/screens/drumSheetPages/sheetXmlDataTemp.dart'
    as pattern_info_default;
import 'package:flutter/material.dart';
// ignore: depend_on_referenced_packages
import 'package:path_provider/path_provider.dart';
import 'package:audioplayers/audioplayers.dart' as ap;
import 'package:capstone_2025/widgets/drum_recording_widget.dart';
import 'package:capstone_2025/screens/mainPages/navigation_screens.dart';
import 'package:flutter/services.dart';
import '../../models/sheet_info.dart';
import '../../models/cursor.dart';
import '../drumSheetPages/widgets/cursor_widget.dart';
import '../drumSheetPages/playback_controller.dart';
import '../../services/osmd_service.dart';
import 'package:capstone_2025/widgets/innerShadow.dart';
import '../drumSheetPages/widgets/confirmation_dialog.dart';
import 'package:capstone_2025/screens/drumSheetPages/widgets/confirmation_dialog.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import '../../services/api_func.dart';
import './practice_result_PP.dart';

/// MenuController에 toggle()을 추가하는 확장 메서드
extension MenuControllerToggle on MenuController {
  void toggle() => isOpen ? close() : open();
}

// 패턴 및 필인 시작 화면
class PatternFillScreen extends StatelessWidget {
  final int index;

  const PatternFillScreen({super.key, required this.index});

  @override
  Widget build(BuildContext context) {
    return CountdownPage(index: index);
  }
}

// 실제 기능을 담당하는 StatefulWidget
class CountdownPage extends StatefulWidget {
  final int index;

  const CountdownPage({super.key, required this.index});

  @override
  State<CountdownPage> createState() => _CountdownPageState();
}

class _CountdownPageState extends State<CountdownPage>
    with SingleTickerProviderStateMixin {
// ===== 악보 및 XML 관련 변수 =====
  late String xmlDataString;
  int _beatsPerMeasure = 4;
  int _totalMeasures = 1;
  double _bpm = 60.0;

  // ===== 재생 및 마디 관련 변수 =====
  final int _currentMeasure = 0; // 녹음 마디 (0-based)
  final int _currentMeasureOneBased = 0; // 채점용 마디 (1-based)

  // 상태 변수 선언
  bool _isPlaying = false;
  bool _showPracticeMessage = false;
  bool _playbackComplete = false; // 연주 상태 완료 추적
  double _currentPosition = 0.0;
  double _totalDuration = 0.0;
  String _currentSpeed = '1.0x';
  List<dynamic> _detectedOnsets = [];

  // DrumRecordingWidget에 대한 키 생성
  final GlobalKey<DrumRecordingWidgetState> _drumRecordingKey = GlobalKey();

  // 객체들
  late ap.AudioPlayer _audioPlayer;
  late AnimationController _overlayController;
  late Animation<double> _overlayAnimation;

  // 타이머들
  Timer? _practiceMessageTimer;
  Timer? _positionUpdateTimer;

  // 스트림 구독들
  StreamSubscription? _playerStateSubscription;
  StreamSubscription? _playerCompleteSubscription;
  StreamSubscription? _positionSubscription;

  // 악보 관련
  late PlaybackController playbackController;
  late OSMDService osmdService;
  bool _isControllerInitialized = false;

  // 배속 설정 메뉴 컨트롤러
  late MenuController _speedMenuController;

  final _storage = const FlutterSecureStorage();

// 서버에서 받아온 패턴 WAV를 저장한 로컬 경로
  String? _patternAudioPath;

  // 채점 결과 처리
  final List<Map<String, dynamic>> _beatGradingResults = [];

  // 페이지 로딩 시 API 호출 후 받아오는 데이터
  String patternName = 'Default Pattern Name';
  String patternInfo = pattern_info_default.temp;

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    if (!_isControllerInitialized) {
      final imageHeight = MediaQuery.of(context).size.height * 0.27;
      playbackController = PlaybackController(imageHeight: imageHeight)
        ..onProgressUpdate = (progress) {
          setState(() {
            _currentPosition = progress * _totalDuration;
          });
        }
        ..onPlaybackStateChange = (isPlaying) {
          setState(() {
            _isPlaying = isPlaying;
            // 재생이 멈추고, 끝까지 도달했으면 완료 플래그 세팅
            if (!isPlaying &&
                playbackController.currentDuration >=
                    playbackController.totalDuration) {
              _playbackComplete = true;
              _currentPosition = _totalDuration;
            }
          });
        }
        ..onCountdownUpdate = (count) {
          setState(() {});
          if (count == 0) {
            _drumRecordingKey.currentState?.startRecording();
          }
        };
      _isControllerInitialized = true;
    }
  }

  @override
  void initState() {
    super.initState();

    // 오버레이 애니메이션 초기화
    _overlayController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 500),
    );
    _overlayAnimation =
        Tween<double>(begin: 0.0, end: 1.0).animate(_overlayController);

    // 오디오 플레이어 초기화
    _audioPlayer = ap.AudioPlayer();
    _setupAudioListeners();

    // 서버에서 패턴 WAV를 미리 받아오기
    _downloadPatternWav();

    // OSMDService 생성 (데이터 로드 완료 콜백만 정의)
    osmdService = OSMDService(onDataLoaded: _onDataLoaded);

    // 페이지가 로드될 때 API 호출
    _fetchData().then((_) {
      // base64 → bytes → utf8 string
      final xmlBytes = base64Decode(patternInfo);
      final xmlString = utf8.decode(xmlBytes);
      // 악보 렌더 실행
      osmdService.startOSMDService(
        xmlData: utf8.encode(xmlString),
        pageWidth: 1080,
      );
    });
  }

  Future<void> _downloadPatternWav() async {
    final patternId = widget.index.toString();
    print('⬇️ [Download] 패턴 $patternId WAV 다운로드 시작');

    try {
      final resData = await getHTTP("/patterns/$patternId/wavs", {});
      print('🔍 [Download] resData 전체: $resData');

      if (resData['errMessage'] != null) {
        print('❌ [Download] 서버 에러: ${resData['errMessage']}');
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('WAV 다운로드 실패: ${resData['errMessage']}')),
        );
        return;
      }

      final b64 = resData['body']['patternWav'] as String;

      if (b64.isEmpty) {
        print('❌ [Download] patternWav 필드가 비어있음!');
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('WAV 데이터가 없습니다.')),
        );
        return;
      }

      final bytes = base64Decode(b64);
      final dir = await getTemporaryDirectory();
      final file = File('${dir.path}/pattern_$patternId.wav');
      await file.writeAsBytes(bytes, flush: true);
      print('✅ [Download] 파일 저장 완료: ${file.path}');
      setState(() {
        _patternAudioPath = file.path;
      });
    } catch (e) {
      print("❌ [Download] 예외 발생: $e");
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('WAV 다운로드 중 예외 발생: $e')),
      );
    }
  }

  // OSMDService 초기화할 때 onDataLoaded 연결
  Future<void> _onDataLoaded({
    required Uint8List base64Image,
    required Map<String, dynamic> json,
    required double bpm,
    required double canvasWidth,
    required double canvasHeight,
    required List<dynamic> lineBounds,
    required int totalMeasures,
  }) async {
    try {
      final List<Uint8List> lineImages = (json['lineImages'] as List<dynamic>)
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
        playbackController.rawCursorList = sheetInfo.cursorList;

        playbackController.currentLineImage =
            lineImages.isNotEmpty ? lineImages[0] : null;
      });
    } catch (e, st) {
      debugPrint('🔴 onDataLoaded error: $e\n$st');
    }
  }

  // API 호출 함수
  Future<void> _fetchData() async {
    print("API 호출 시작");

    String? patternId = widget.index.toString();

    // JSON 데이터 정의
    final Map<String, String> queryParam = {
      // API를 통해 전달할 param
      "patternId ": patternId,
    };

    Map<String, dynamic> resData =
        await getHTTP("/patterns/$patternId", queryParam);

    if (resData['errMessage'] == null) {
      patternName = resData['body']['patternName'] ??
          'Default Pattern Name'; // null일 경우 기본값 설정
      patternInfo = resData['body']['patternInfo'] ??
          pattern_info_default.temp; // null일 경우 기본값 설정
    } else {
      print("패턴 정보 요청 실패: ${resData['errMessage']}");
    }
  }

  void _setupAudioListeners() {
    // 오디오 재생 상태 리스너
    _playerStateSubscription =
        _audioPlayer.onPlayerStateChanged.listen((state) {
      if (!mounted) return;
      setState(() {
        _isPlaying = state == ap.PlayerState.playing;
      });
      print('▶️ PlayerState: $state');
    });

    // 오디오 재생 위치 리스너
    final positionSubscription =
        _audioPlayer.onPositionChanged.listen((position) {
      if (!mounted) return;
      setState(() {
        // 밀리초 단위로
        _currentPosition = position.inMilliseconds.toDouble();
      });
    });

    // 오디오 총 길이 리스너
    _audioPlayer.onDurationChanged.listen((duration) {
      if (!mounted) return;
      setState(() {
        _totalDuration = duration.inMilliseconds.toDouble();
      });
    });
  }

  // 시범 연주를 재생하는 함수
  Future<void> _startAudio() async {
    if (!mounted) return;

    // 초기화 + 재생 상태 + 연습 메시지 표시까지 모두 처리
    setState(() {
      _currentPosition = 0.0;
      _playbackComplete = false;
      _isPlaying = true;
      _showPracticeMessage = true;
    });

    _overlayController.forward();

    // 1초 후 메시지 숨기기
    await Future.delayed(const Duration(milliseconds: 1000));

    if (!mounted) return;
    _overlayController.reverse().then((_) {
      if (mounted) setState(() => _showPracticeMessage = false);
    });

    // 다운로드가 아직 안 끝났으면 최대 5초까지 기다려 봄
    if (_patternAudioPath == null) {
      print('⏳ 아직 다운로드 중… 최대 5초 대기');
      await Future.any([
        Future.delayed(const Duration(seconds: 5)),
        Future.doWhile(() async {
          await Future.delayed(const Duration(milliseconds: 200));
          return _patternAudioPath == null;
        }),
      ]);
    }

    // 시범 연주 오디오 재생
    if (_patternAudioPath != null) {
      print('🎧 Playing server-fetched WAV: $_patternAudioPath');
      await _audioPlayer.play(
        ap.DeviceFileSource(_patternAudioPath!),
      );
    } else {
      const errMsg = '❌ 서버에서 WAV 파일을 불러오지 못했습니다.';
      print(errMsg);
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text(errMsg)),
      );
      return; // 함수 종료
    }

    StreamSubscription<Duration>? oneShotSub;
    oneShotSub = _audioPlayer.onDurationChanged.listen((d) {
      debugPrint('🎵 시범 연주 총 길이: ${d.inSeconds}초');
      oneShotSub?.cancel();
    });

    // 시범 연주가 끝나면 PlaybackController 카운트다운 + 녹음 시작
    _playerCompleteSubscription =
        _audioPlayer.onPlayerComplete.listen((_) async {
      if (!mounted) return;

      _playerCompleteSubscription?.cancel();

      try {
        // 오디오가 stopped 상태가 아니면 seek 시도
        if (_audioPlayer.state != ap.PlayerState.stopped) {
          await _audioPlayer
              .seek(Duration.zero)
              .timeout(const Duration(seconds: 1));
        }
      } catch (e, stack) {
        debugPrint('seek 예외 발생(무시해도 됨): $e\n$stack');
      }

      setState(() => _currentPosition = 0.0);

      // 카운트다운 UI → 3-2-1 → 시트 재생
      playbackController.showCountdownAndStart();
    });
  }

  // 오디오 재생 위치 이동하는 함수
  void _seekAudio(double position) async {
    if (!mounted) return;
    await _audioPlayer.seek(Duration(seconds: position.toInt())); // 재생 위치 이동
  }

  // 재생 속도 선택 버튼을 만드는 위젯
  Widget _buildSpeedButton(String speed, bool isSelected) {
    return Material(
      color: Colors.transparent,
      child: InkWell(
        onTap: () {
          double newSpeed = switch (speed) {
            '0.5x' => 0.5,
            '1x' => 1.0,
            '1.5x' => 1.5,
            '2x' => 2.0,
            _ => 1.0,
          };
          if (mounted) {
            setState(() {
              _audioPlayer.setPlaybackRate(newSpeed); // 오디오 재생 속도 변경
              _currentSpeed = speed; // 현재 속도 업데이트
            });
          }
          Navigator.of(context).pop(); // 속도 변경 후 팝업 닫기
        },
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 5),
          child: Text(
            speed,
            style: TextStyle(
              color: isSelected
                  ? const Color(0xFFD97D6C)
                  : const Color(0xFF646464),
              fontWeight: FontWeight.w800,
              fontSize: 24,
            ),
          ),
        ),
      ),
    );
  }

  Future<String?> fetchPracticeIdentifier() async {
    // 1) 토큰 읽기
    final token = await _storage.read(key: 'access_token');

    // 2) POST 호출
    final response = await postHTTP(
      '/audio/practice',
      null,
      reqHeader: {'authorization': token ?? ''},
    );

    // 3) 결과 처리
    if (response['errMessage'] == null) {
      return response['body'] as String;
    } else {
      print('Identifier 요청 실패: ${response['errMessage']}');
      return null;
    }
  }

  // 1차 채점 결과 처리
  void _handleScoringResult(Map<String, dynamic> scoringResult) {
    final measureNumber = scoringResult['measureNumber'];
    final answerOnsetPlayed =
        List<bool>.from(scoringResult['answerOnsetPlayed']);
    final measureIndex = int.parse(measureNumber) - 1;

    // 틀린 박자 인덱스만 골라내기
    final missedNotesIndices = <int>[];
    for (int i = 0; i < answerOnsetPlayed.length; i++) {
      if (!answerOnsetPlayed[i]) missedNotesIndices.add(i);
    }

    // PlaybackController 에 커서 추가
    playbackController.addMissedNotesCursor(
      measureIndex: measureIndex,
      missedIndices: missedNotesIndices,
    );
    setState(() {}); // 화면 갱신
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

  // 최종 채점 결과 적용 및 결과 화면 이동
  void _applyGradingResults() {
    print("✅ 1차 채점 완료: measureNumbers = "
        "${_beatGradingResults.map((m) => m['measureNumber']).toList()}");
    final initialBeatScore = computeScoreFrom1stGrading(_beatGradingResults);

    // 2초 딜레이 후 결과창으로 이동
    Future.delayed(const Duration(seconds: 2), () {
      Navigator.of(context).pushReplacement(
        MaterialPageRoute(
          builder: (_) => PracticeResultPP(
            idx: widget.index,
            score: initialBeatScore,
            xmlDataString: patternInfo,
            practiceInfo: practiceInfo,
          ),
        ),
      );
    });
  }

  void _onWsGradingMessage(Map<String, dynamic> msg) {
    _beatGradingResults.add(msg);
    // 모든 마디 (4마디) 데이터가 수신되면 결과 처리
    if (_beatGradingResults.length == 4) {
      print(
          "🗒️ 전체 마디 데이터 수신 완료: ${_beatGradingResults.map((m) => m['measureNumber']).toList()}");
      _applyGradingResults();
    }
  }

  @override
  void dispose() {
    _practiceMessageTimer?.cancel();
    _positionUpdateTimer?.cancel();
    _playerStateSubscription?.cancel();
    _playerCompleteSubscription?.cancel();
    _positionSubscription?.cancel();

    // 오디오플레이어 정리
    _audioPlayer.dispose();

    // 오버레이 애니메이션 컨트롤러 정리
    _overlayController.dispose();

    // (중요) OSMDService 등 서버 리소스 정리
    osmdService.dispose(); // 만약 dispose 메서드가 있다면

    // DrumRecordingWidget의 리소스도 정리 필요시
    _drumRecordingKey.currentState?.dispose();

    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final screenH = MediaQuery.of(context).size.height;
    final screenW = MediaQuery.of(context).size.width;
    if (playbackController.sheetInfo == null) {
      return const Center(child: CircularProgressIndicator());
    } // 악보 불러올 때까지 로딩 추가 ..?

    return Scaffold(
      backgroundColor: const Color(0xFFF5F5F5), // 배경색
      body: Stack(
        children: [
          SafeArea(
            child: Column(
              children: [
                const SizedBox(height: 10), // 상단 여백

                // 상단 영역: 홈 버튼 + 제목 + 속도 변경 버튼
                Padding(
                  padding:
                      const EdgeInsets.symmetric(horizontal: 46, vertical: 10),
                  child: SizedBox(
                    height: 60,
                    child: Row(children: [
                      SizedBox(
                        width: 60,
                        height: 30,
                        child: Center(
                          child:
                              // 홈 버튼
                              IconButton(
                            padding: EdgeInsets.zero,
                            constraints: const BoxConstraints(),
                            iconSize: 30,
                            icon: const Icon(Icons.home_filled,
                                color: Color(0xff646464)),
                            onPressed: () {
                              // Show the confirmation dialog
                              showDialog(
                                context: context,
                                builder: (context) => ConfirmationDialog(
                                  message: "메인으로 이동하시겠습니까?",
                                  onConfirm: () {
                                    // The logic when the user confirms
                                    // Your existing home button behavior (like stopping playback and going home)
                                    Navigator.of(context)
                                        .pop(); // Close the dialog
                                    // Proceed with your home navigation logic (similar to DrumSheetPlayer)
                                    if (_isPlaying) _audioPlayer.stop();
                                    final drumRecordingState =
                                        _drumRecordingKey.currentState;
                                    if (drumRecordingState != null &&
                                        drumRecordingState.isRecording) {
                                      drumRecordingState.stopRecording();
                                    }
                                    _playerStateSubscription?.cancel();
                                    _playerCompleteSubscription?.cancel();
                                    _practiceMessageTimer?.cancel();
                                    _positionUpdateTimer?.cancel();

                                    WidgetsBinding.instance
                                        .addPostFrameCallback((_) {
                                      final navigationScreensState =
                                          context.findAncestorStateOfType<
                                              NavigationScreensState>();
                                      if (navigationScreensState != null &&
                                          navigationScreensState.mounted) {
                                        navigationScreensState.setState(() {
                                          navigationScreensState.selectedIndex =
                                              2;
                                        });
                                      }
                                      if (Navigator.canPop(context)) {
                                        Navigator.of(context).pop();
                                      }
                                    });
                                  },
                                  onCancel: () {
                                    // Close the dialog if user cancels
                                    Navigator.of(context).pop();
                                  },
                                ),
                              );
                            },
                          ),
                        ),
                      ),

                      // 타이틀 Container
                      Expanded(
                        child: Center(
                          child: InnerShadow(
                            shadowColor:
                                const Color.fromARGB(255, 238, 159, 145)
                                    .withValues(alpha: 0.5),
                            blur: 6,
                            offset: Offset(0, 0),
                            borderRadius: BorderRadius.circular(30),
                            child: Builder(builder: (context) {
                              final screenW = MediaQuery.of(context).size.width;
                              return Container(
                                constraints:
                                    BoxConstraints(maxWidth: screenW * 0.45),
                                height: 54,
                                padding:
                                    const EdgeInsets.symmetric(horizontal: 28),
                                alignment: Alignment.center,
                                decoration: BoxDecoration(
                                  color: const Color(0xFFC76A59),
                                  borderRadius: BorderRadius.circular(30),
                                  border: Border.all(
                                    color: const Color(0xFFB95D4C), // 테두리 색
                                    width: 4,
                                  ),
                                ),
                                child: Stack(
                                  alignment: Alignment.center,
                                  children: [
                                    // 아래: 테두리용 텍스트
                                    Text(
                                      patternName,
                                      maxLines: 1,
                                      overflow: TextOverflow.ellipsis,
                                      style: TextStyle(
                                        fontSize: 25,
                                        fontWeight: FontWeight.bold,
                                        foreground: Paint()
                                          ..style = PaintingStyle.stroke
                                          ..strokeWidth = 5
                                          ..color =
                                              const Color(0xFFB95D4C), // 테두리 색
                                      ),
                                    ),
                                    // 위: 흰색 채우기 텍스트
                                    Text(
                                      patternName,
                                      maxLines: 1,
                                      overflow: TextOverflow.ellipsis,
                                      style: const TextStyle(
                                        fontSize: 25,
                                        fontWeight: FontWeight.bold,
                                        color: Colors.white, // 내부 색
                                      ),
                                    ),
                                  ],
                                ),
                              );
                            }),
                          ),
                        ),
                      ),

                      // 오른쪽 상단: 재생 속도 조정 버튼
                      MenuAnchor(
                        style: MenuStyle(
                          alignment: Alignment.bottomLeft,
                          shadowColor:
                              WidgetStatePropertyAll(Colors.transparent),
                          padding: WidgetStatePropertyAll(EdgeInsets.zero),
                          backgroundColor:
                              WidgetStatePropertyAll(Colors.transparent),
                        ),
                        menuChildren: [
                          ConstrainedBox(
                            constraints: BoxConstraints(maxWidth: 500),
                            child: Container(
                              margin: const EdgeInsets.fromLTRB(0, 10, 20, 0),
                              padding: const EdgeInsets.symmetric(
                                  horizontal: 18, vertical: 12),
                              decoration: BoxDecoration(
                                color: Colors.white,
                                borderRadius: BorderRadius.circular(18),
                                border: Border.all(
                                    color: const Color(0xFFDFDFDF), width: 2),
                                boxShadow: [
                                  BoxShadow(
                                    color: Colors.black.withOpacity(0.08),
                                    blurRadius: 6,
                                    offset: Offset(0, 4),
                                  ),
                                ],
                              ),
                              child: Row(
                                  mainAxisSize: MainAxisSize.min,
                                  children: [0.5, 1.0, 1.5, 2.0].map((s) {
                                    final label = '${s.toStringAsFixed(1)}x';
                                    final isSelected =
                                        playbackController.speed == s;
                                    return Padding(
                                      padding: EdgeInsets.only(
                                          left: 15, right: s == 2.0 ? 0 : 15),
                                      child: GestureDetector(
                                        onTap: () {
                                          // 재생 중일 때는 배속 변경 못하도록 함
                                          if (!playbackController.isPlaying) {
                                            // OSMD(악보) 재생 속도 변경
                                            playbackController.setSpeed(s);
                                            // 오디오 재생 속도 변경
                                            _audioPlayer.setPlaybackRate(s);
                                            setState(() {
                                              _currentSpeed = label;
                                            });
                                          }
                                          _speedMenuController.close();
                                        },
                                        child: Text(
                                          label,
                                          style: TextStyle(
                                            fontSize: 20,
                                            fontWeight: FontWeight.bold,
                                            color: isSelected
                                                ? const Color(0xffD97D6C)
                                                : const Color(0xff646464),
                                          ),
                                        ),
                                      ),
                                    );
                                  }).toList()),
                            ),
                          ),
                        ],
                        builder: (context, controller, child) {
                          _speedMenuController = controller;
                          return GestureDetector(
                            onTap: () => controller.toggle(),
                            child: SizedBox(
                              width: 60,
                              height: 30,
                              child: Center(
                                child: Text(
                                  _currentSpeed,
                                  style: const TextStyle(
                                    color: Color(0xFF646464),
                                    fontWeight: FontWeight.bold,
                                    fontSize: 25,
                                  ),
                                ),
                              ),
                            ),
                          );
                        },
                      ),
                    ]),
                  ),
                ),

                Spacer(flex: 1),

                // 본문 영역 (악보 + 온셋 수 표시)
                Container(
                  padding: const EdgeInsets.fromLTRB(20, 0, 0, 10),
                  width: screenW * 0.93,
                  height: screenH * 0.3,
                  decoration: BoxDecoration(
                    color: Colors.white,
                    borderRadius: BorderRadius.circular(12),
                    boxShadow: [
                      BoxShadow(
                        color: Colors.black.withOpacity(0.08),
                        blurRadius: 6,
                        offset: Offset(0, 4),
                      ),
                    ],
                  ),
                  child: Stack(
                    children: [
                      ClipRRect(
                        borderRadius: BorderRadius.circular(5),
                        child: LayoutBuilder(
                          builder: (context, constraints) {
                            final boxW = constraints.maxWidth;
                            final boxH = constraints.maxHeight; // 박스 높이

                            return Stack(
                              children: [
                                // 틀린 박자: 회색 커서
                                for (final missed in playbackController
                                    .missedCursors
                                    .where((c) =>
                                        c.lineIndex ==
                                        playbackController.currentPage))
                                  CursorWidget(
                                    cursor: missed,
                                    imageWidth: boxW,
                                    height: boxH,
                                    decoration: BoxDecoration(
                                      color: Colors.grey.withOpacity(0.6),
                                      borderRadius: BorderRadius.circular(4),
                                    ),
                                  ),
                                // 재생 커서
                                if (playbackController.currentDuration >
                                        Duration.zero ||
                                    playbackController.isPlaying ||
                                    playbackController.currentDuration >=
                                        playbackController.totalDuration)
                                  CursorWidget(
                                    cursor: playbackController.currentCursor,
                                    imageWidth: boxW,
                                    height: boxH,
                                  ),
                                // 악보 이미지
                                if (playbackController.currentLineImage != null)
                                  Image.memory(
                                    width: boxW,
                                    height: boxH * 1.5,
                                    playbackController.currentLineImage!,
                                    fit: BoxFit.fill,
                                    alignment: Alignment.topCenter,
                                  ),
                              ],
                            );
                          },
                        ),
                      ),
                      // 감지된 온셋 수 표시
                      if (_detectedOnsets.isNotEmpty)
                        Positioned(
                          bottom: 10,
                          left: 10,
                          child: Container(
                            padding: const EdgeInsets.all(8),
                            decoration: BoxDecoration(
                              color: Colors.black.withValues(alpha: 0.7),
                              borderRadius: BorderRadius.circular(8),
                            ),
                            child: Text(
                              '감지된 온셋: ${_detectedOnsets.length}개',
                              style: const TextStyle(
                                color: Colors.white,
                                fontWeight: FontWeight.bold,
                              ),
                            ),
                          ),
                        ),
                    ],
                  ),
                ),
                Spacer(flex: 1),
                // 하단 영역 (재생/녹음 버튼 + 녹음 상태 표시)
                Center(
                  child: Container(
                    width: screenW * 0.55,
                    padding: const EdgeInsets.fromLTRB(32, 20, 32, 20),
                    decoration: BoxDecoration(
                      color: const Color(0xFFE1E1E1),
                      borderRadius: BorderRadius.circular(20),
                    ),
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      crossAxisAlignment: CrossAxisAlignment.stretch,
                      children: [
                        // 오디오 재생 위치 표시하는 슬라이더
                        Container(
                          height: 5,
                          decoration: BoxDecoration(
                            color: Colors.white,
                            borderRadius: BorderRadius.circular(20),
                          ),
                          child: FractionallySizedBox(
                            alignment: Alignment.centerLeft,
                            widthFactor: (_totalDuration > 0)
                                ? (_currentPosition / _totalDuration)
                                    .clamp(0.0, 1.0)
                                : 0.0,
                            child: ClipRRect(
                              borderRadius: BorderRadius.circular(20),
                              child: Container(
                                color: const Color(0xFFD97D6C),
                              ),
                            ),
                          ),
                        ),
                        const SizedBox(height: 16),

                        // 재생 / 일시정지 / 리셋 버튼
                        Center(
                          child: GestureDetector(
                            onTap: () {
                              if (_showPracticeMessage)
                                return; // 안내 오버레이 중엔 터치 무시

                              // 완료 상태 → 리셋 다이얼로그
                              if (_playbackComplete) {
                                showDialog(
                                  context: context,
                                  barrierDismissible: true,
                                  builder: (_) => ConfirmationDialog(
                                    message: "처음부터 다시 연주하시겠습니까?",
                                    onConfirm: () {
                                      Navigator.of(context).pop();

                                      // 완전 초기화
                                      playbackController.resetToStart();
                                      _audioPlayer.stop();
                                      // 녹음기록도 초기화
                                      _drumRecordingKey.currentState
                                          ?.stopRecording();
                                      setState(() {
                                        _playbackComplete = false;
                                        _currentPosition = 0.0;
                                        _detectedOnsets = [];
                                      });
                                    },
                                    onCancel: () {
                                      Navigator.of(context).pop();
                                    },
                                  ),
                                );
                                return;
                              }
                              // 재생 중일 땐 아무 동작 안 함
                              if (playbackController.isPlaying) return;

                              // 재생 시작
                              if (_currentPosition == 0.0) {
                                _startAudio(); // 시범 연주
                              }
                            },
                            child: _playbackComplete
                                ? Container(
                                    width: 48,
                                    height: 48,
                                    decoration: BoxDecoration(
                                      shape: BoxShape.circle,
                                      color: Colors.white,
                                      border: Border.all(
                                        color: const Color(0xFFDFDFDF),
                                        width: 2,
                                      ),
                                    ),
                                    child: const Icon(
                                      Icons.replay,
                                      size: 40,
                                      color: Color(0xffD97D6C),
                                    ),
                                  )
                                : (_isPlaying || playbackController.isPlaying)
                                    ? Container(
                                        width: 48,
                                        height: 48,
                                        decoration: BoxDecoration(
                                          shape: BoxShape.circle,
                                          color: Colors.white,
                                          border: Border.all(
                                            color: const Color(0xFFDFDFDF),
                                            width: 2,
                                          ),
                                        ),
                                        child: const Icon(
                                          Icons.pause,
                                          size: 40,
                                          color: Color(0xffD97D6C),
                                        ),
                                      )
                                    : Container(
                                        width: 48,
                                        height: 48,
                                        decoration: const BoxDecoration(
                                          shape: BoxShape.circle,
                                          color: Color(0xffD97D6C),
                                        ),
                                        child: const Icon(
                                          Icons.play_arrow,
                                          size: 40,
                                          color: Colors.white,
                                        ),
                                      ),
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ],
            ),
          ),

          // 오버레이 - 시범 연주 안내 문구
          if (_showPracticeMessage)
            FadeTransition(
              opacity: _overlayAnimation,
              child: Container(
                color: Colors.black.withValues(alpha: 0.9),
                alignment: Alignment.center,
                child: const Text(
                  '시범 연주를 시작하겠습니다',
                  style: TextStyle(
                    color: Colors.white,
                    fontSize: 24,
                    fontWeight: FontWeight.bold,
                  ),
                ),
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

          // DrumRecordingWidget 추가
          // Offstage(
          //   offstage: true, // UI를 화면에 표시하지 않음
          //   child: DrumRecordingWidget(
          //     key: _drumRecordingKey,
          //     playbackController: playbackController,
          //     title: 'Basic Pattern ${widget.index}',
          //     xmlDataString: patternInfo,
          //     userSheetId: widget.index,
          //     audioFilePath: 'assets/sounds/test_pattern.wav',
          //     fetchPracticeIdentifier:
          //         fetchPracticeIdentifier, // identifier 요청 함수
          //     onMusicXMLParsed: (info) {
          //       _drumRecordingKey.currentState?.setMeasureInfo(info);
          //     },
          //     onRecordingComplete: (onsets) {
          //       setState(() {
          //         _detectedOnsets = onsets;
          //       });
          //     },
          //     onOnsetsReceived: (onsets) {
          //       setState(() {
          //         _detectedOnsets = onsets;
          //       });
          //     },
          //     onGradingResult: (msg) {
          //       _handleScoringResult(msg); // 1) 즉시 화면에 틀린 박자 커서 표시
          //       _onWsGradingMessage(msg); // 2) 리스트에 쌓아서, 마지막에 전체 점수 계산
          //     },

          //   ),
          // ),

          // DrumRecordingWidget 추가 (보이지 않지만 기능 사용)
          Offstage(
            offstage: true, // UI를 화면에 표시하지 않음
            child: DrumRecordingWidget(
              key: _drumRecordingKey,
              patternId: widget.index,
              title: 'Basic Pattern ${widget.index}',
              xmlDataString: patternInfo,
              audioFilePath: _patternAudioPath ?? '',
              playbackController: playbackController,
              fetchPracticeIdentifier:
                  fetchPracticeIdentifier, // identifier 요청 함수
              onRecordingComplete: (onsets) {
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
              // onMusicXMLParsed: (info) {
              //   _drumRecordingKey.currentState?.setMeasureInfo(info);
              // },

              onOnsetsReceived: (onsets) {
                setState(() {
                  _detectedOnsets = onsets;
                });
              },
              onGradingResult: (msg) {
                _handleScoringResult(msg); // 1) 즉시 화면에 틀린 박자 커서 표시
                _onWsGradingMessage(msg); // 2) 리스트에 쌓아서, 마지막에 전체 점수 계산
              },
              // playbackController: playbackController, //playbackController 전달
              // fetchPracticeIdentifier: fetchPracticeIdentifier,
            ),
          ),
        ],
      ),
    );
  }
}
