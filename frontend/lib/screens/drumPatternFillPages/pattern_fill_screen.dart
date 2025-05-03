import 'dart:io';
import 'dart:async';
import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';
import 'package:audioplayers/audioplayers.dart' as ap;
import 'package:flutter_sound/flutter_sound.dart' as fs;
import 'package:flutter_sound/public/flutter_sound_recorder.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import 'package:stomp_dart_client/stomp_dart_client.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:capstone_2025/screens/mainPages/navigation_screens.dart';

// 패턴 및 필인 시작 화면
class PatternFillScreen extends StatelessWidget {
  final String title;

  const PatternFillScreen({super.key, required this.title});

  @override
  Widget build(BuildContext context) {
    return CountdownPage(title: title);
  }
}

// 실제 기능을 담당하는 StatefulWidget
class CountdownPage extends StatefulWidget {
  final String title;

  const CountdownPage({super.key, required this.title});

  @override
  State<CountdownPage> createState() => _CountdownPageState();
}

class _CountdownPageState extends State<CountdownPage>
    with SingleTickerProviderStateMixin {
  // 상태 변수 선언
  int countdown = 3;
  bool isCountingDown = false;
  bool _isPlaying = false;
  bool _isRecording = false;
  bool _showPracticeMessage = false;
  bool _webSocketConnected = false;

  double _currentPosition = 0.0;
  double _totalDuration = 0.0;

  String _currentSpeed = '1x';
  String _recordingStatusMessage = '';
  String _userEmail = '';
  String? _recordingPath; // 녹음 파일 경로

  List<dynamic> _detectedOnsets = [];

  // 객체들
  late ap.AudioPlayer _audioPlayer; // 오디오
  late fs.FlutterSoundRecorder _recorder;
  late AnimationController _overlayController; // 애니메이션
  late Animation<double> _overlayAnimation;
  late StompClient _stompClient; // 웹소켓 클라이언트

  // 저장소
  final _storage = const FlutterSecureStorage();

  // 타이머들
  Timer? _countdownTimer;
  Timer? _practiceMessageTimer;
  Timer? _positionUpdateTimer;
  Timer? _recordingDataTimer;

  // 스트림 구독들
  StreamSubscription? _playerStateSubscription;
  StreamSubscription? _playerCompleteSubscription;
  StreamSubscription? _positionSubscription;
  StreamSubscription<fs.RecordingDisposition>? _recorderSubscription;

  // 웹소켓 재연결 관련 변수
  int _reconnectAttemps = 0;
  final int _maxReconnectAttempts = 5;

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

    // 필요한 데이터 초기화
    _initializeData();
  }

  Future<void> _initializeData() async {
    // 저장된 이메일 불러오기 (없으면 기본값)
    _userEmail = await _storage.read(key: 'user_email') ?? 'test@example.com';

    _audioPlayer = ap.AudioPlayer();
    _recorder = fs.FlutterSoundRecorder();

    await _initRecorder();

    _setupAudioListeners();
    _setupWebSocket();
  }

  Future<void> _initRecorder() async {
    // 마이크 권한 요청
    var status = await Permission.microphone.request();
    if (status != PermissionStatus.granted) {
      throw RecordingPermissionException('마이크 권한이 부여되지 않았습니다.');
    }

    await _recorder.openRecorder();

    // 녹음 파일 저장 경로 설정
    final appDocDir = await getApplicationDocumentsDirectory();
    _recordingPath = '${appDocDir.path}/drum_performance.wav';
  }

  void _setupAudioListeners() {
    // 오디오 재생 상태 리스너
    _playerStateSubscription =
        _audioPlayer.onPlayerStateChanged.listen((state) {
      if (!mounted) return;
      setState(() {
        _isPlaying = state == ap.PlayerState.playing;
      });
    });

    // 오디오 재생 위치 리스너
    _positionSubscription = _audioPlayer.onPositionChanged.listen((position) {
      if (!mounted) return;
      setState(() {
        _currentPosition = position.inSeconds.toDouble();
      });
    });

    // 오디오 총 길이 리스너
    _audioPlayer.onDurationChanged.listen((duration) {
      if (!mounted) return;
      setState(() {
        _totalDuration = duration.inSeconds.toDouble();
      });
    });
  }

  void _setupWebSocket() {
    // WebSocket 설정
    _stompClient = StompClient(
      config: StompConfig.sockJS(
        url: 'http://10.0.2.2:28080/ws/audio',
        onConnect: (StompFrame frame) {
          print('✅ WebSocket 연결 완료!');
          _webSocketConnected = true;
          _reconnectAttemps = 0; // 연결 성공했으니 재시도 카운트 초기화
          _subscribeToTopic();
        },
        beforeConnect: () async => print('🌐 WebSocket 연결 시도 중...'),
        // 오류 발생했을 때
        onWebSocketError: (dynamic error) {
          print('❌ WebSocket 오류 발생: $error');
          _retryWebSocketConnect();
        },
        onDisconnect: (frame) {
          print('🔌 WebSocket 연결 끊어짐');
          setState(() {
            _webSocketConnected = false;
          });
        },
        stompConnectHeaders: {},
      ),
    );
    // WebSocket 연결 시도
    _stompClient.activate();
  }

  // 연결 후 구독
  void _subscribeToTopic() {
    _stompClient.subscribe(
      destination: '/topic/onset/$_userEmail',
      callback: (frame) {
        if (frame.body != null) {
          final response = json.decode(frame.body!);
          print('📦 WebSocket 데이터 수신 완료: $response');

          if (response.containsKey('onsets')) {
            setState(() {
              _detectedOnsets = response['onsets'];
            });
            print('🎯 감지된 온셋 수: ${response['onsets']}');
          }
        } else {
          print('⚠️ 빈 WebSocket 프레임 수신');
        }
      },
    );
  }

  // 웹소켓 연결 실패 시 재시도하는 함수
  void _retryWebSocketConnect() {
    if (_reconnectAttemps < _maxReconnectAttempts) {
      _reconnectAttemps++;
      Future.delayed(const Duration(seconds: 3), () {
        print(
            '🔁 WebSocket 재연결 시도 ($_reconnectAttemps/$_maxReconnectAttempts)...');
        _stompClient.activate();
      });
    } else {
      print('❌ WebSocket 재연결 실패 - 최대 시도 초과');
    }
  }

  // 시범 연주를 재생하는 함수
  Future<void> _startAudio() async {
    if (!mounted) return;

    // 시범 연주 시작 메시지 표시
    setState(() => _showPracticeMessage = true);
    _overlayController.forward(); // 메시지를 페이드 인 애니메이션으로 보여줌

    // 1초 후 메시지 숨기기
    await Future.delayed(const Duration(milliseconds: 1000));

    if (!mounted) return;
    _overlayController.reverse().then((_) {
      if (mounted) setState(() => _showPracticeMessage = false); // 메시지 숨김
    });

    // 메시지가 사라진 후 바로 시범 연주 오디오 재생
    await _audioPlayer.play(ap.AssetSource('test/tom_mix.wav'));

    // 시범 연주가 끝나면 카운트다운 시작
    _playerCompleteSubscription = _audioPlayer.onPlayerComplete.listen((event) {
      if (mounted) _startCountdown(); // 카운트다운 시작
    });
  }

  // 사용자 연주 녹음을 시작하는 함수
  void _startRecording() async {
    if (_isRecording || !mounted) return;

    // WebSocket 연결 확인
    if (!_webSocketConnected) {
      print('❌ 녹음을 시작할 수 없습니다: WebSocket이 연결되지 않았습니다.');
      setState(() => _recordingStatusMessage = 'WebSocket 연결이 필요합니다!');
      return;
    }

    try {
      print("🎙️ 녹음을 시작합니다. 저장 경로: $_recordingPath");
      await _recorder.startRecorder(
        toFile: _recordingPath,
        codec: fs.Codec.pcm16WAV, // wav 형식으로 녹음 저장
        sampleRate: 16000,
        numChannels: 1,
        bitRate: 16000,
      );
      setState(() {
        _isRecording = true;
        _recordingStatusMessage = '녹음이 시작되었습니다.';
      });

      _recordingDataTimer = Timer.periodic(const Duration(seconds: 1), (timer) {
        _sendRecordingData();
      });
    } catch (e) {
      setState(() => _recordingStatusMessage = '녹음 시작 실패: $e');
      print('❌ 녹음 중 오류 발생: $e');
    }
  }

  // 녹음을 중단하는 함수
  Future<void> _stopRecording() async {
    if (!_isRecording || !mounted) return;
    _recordingDataTimer?.cancel(); // 데이터 전송 타이머 중지
    await _recorder.stopRecorder(); // 녹음기 종료

    // 마지막 녹음 데이터 서버로 전송
    _sendRecordingData();

    setState(() {
      _isRecording = false;
      _recordingStatusMessage = '녹음이 완료되었습니다.';
    });

    print('🎙️ 녹음 종료');
  }

  // 녹음된 데이터를 WebSocket을 통해 서버로 전송하는 함수
  Future<void> _sendRecordingData() async {
    if (!_stompClient.connected) {
      print('❌ WebSocket 연결이 되지 않아 데이터 전송 실패');
      return;
    }

    try {
      final file = File(_recordingPath!);
      if (await file.exists()) {
        final base64String = base64Encode(await file.readAsBytes());
        final message = {'email': _userEmail, 'message': base64String};
        print('📤 녹음 데이터 전송: ${DateTime.now()}');

        _stompClient.send(
          destination: '/app/audio/forwarding',
          body: json.encode(message),
          headers: {'content-type': 'application/json'},
        );
        setState(() => _recordingStatusMessage = '녹음 데이터 전송 중...');
      } else {
        print('⚠️ 녹음 파일이 존재하지 않습니다: $_recordingPath');
      }
    } catch (e) {
      print('❌ 녹음 데이터 전송 중 오류 발생: $e');
    }
  }

  // 오디오 재생 위치 이동하는 함수
  void _seekAudio(double position) async {
    if (!mounted) return;
    await _audioPlayer.seek(Duration(seconds: position.toInt())); // 재생 위치 이동
  }

  // 3초 카운트다운 후 녹음 시작하는 함수
  void _startCountdown() {
    if (!mounted) return;
    setState(() {
      isCountingDown = true;
      countdown = 3; // 카운트다운 3초로 시작
    });

    _overlayController.forward(); // 카운트다운 페이드 인

    _countdownTimer = Timer.periodic(const Duration(seconds: 1), (timer) {
      if (!mounted) {
        timer.cancel();
        return;
      }

      if (countdown == 0) {
        timer.cancel();
        _overlayController.reverse().then((_) async {
          if (!mounted) return;
          setState(() => isCountingDown = false);
          _startRecording(); // 카운트다운 종료 후 녹음 시작
        });
      } else {
        setState(() => countdown--);
      }
    });
  }

  // 페이지가 종료될 때 리소스 해제하는 함수
  @override
  void dispose() {
    _countdownTimer?.cancel();
    _practiceMessageTimer?.cancel();
    _positionUpdateTimer?.cancel();
    _recordingDataTimer?.cancel();
    _playerStateSubscription?.cancel();
    _playerCompleteSubscription?.cancel();
    _positionSubscription?.cancel();
    _recorderSubscription?.cancel();

    if (_isRecording) _recorder.stopRecorder();

    _recorder.closeRecorder(); // 녹음기 닫기
    _audioPlayer.dispose(); // 오디오플레이어 정리
    _overlayController.dispose(); // 애니메이션 컨트롤러

    if (_stompClient.connected) _stompClient.deactivate(); // 웹소켓 연결 해제

    super.dispose();
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
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
          child: Text(
            speed,
            style: TextStyle(
              color: isSelected ? const Color(0xFFE5958B) : Colors.black87,
              fontWeight: FontWeight.w500,
              fontSize: 14,
            ),
          ),
        ),
      ),
    );
  }

  // 카운트다운 숫자를 그리는 위젯
  Widget _buildCountdownNumber(int number) {
    final bool isHighlighted = number == countdown;

    return AnimatedOpacity(
      duration: const Duration(milliseconds: 300),
      opacity: isHighlighted ? 1.0 : 0.3,
      child: Text(
        number.toString(),
        style: TextStyle(
          fontSize: 100,
          fontWeight: FontWeight.bold,
          color: isHighlighted ? Colors.red : Colors.white,
          shadows: [
            Shadow(
              offset: const Offset(2, 2),
              blurRadius: 4,
              color: Colors.black.withValues(alpha: 0.5),
            ),
          ],
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFF2F1F3), // 배경색
      body: Stack(
        children: [
          SafeArea(
            child: Column(
              children: [
                const SizedBox(height: 10), // 상단 여백

                // 상단 영역: 홈 버튼 + 제목 + 속도 변경 버튼
                Padding(
                  padding:
                      const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
                  child: Stack(children: [
                    Row(
                      children: [
                        // 홈 버튼
                        IconButton(
                          icon: const Icon(Icons.home_filled),
                          onPressed: () {
                            // 오디오 재생 중이면 정지
                            if (_isPlaying) _audioPlayer.stop();
                            // 녹음 중이면 정지
                            if (_isRecording) _stopRecording();

                            // 리소스 해제
                            _playerStateSubscription?.cancel();
                            _playerCompleteSubscription?.cancel();
                            _countdownTimer?.cancel();
                            _practiceMessageTimer?.cancel();
                            _positionUpdateTimer?.cancel();

                            // 홈화면으로 이동: NavigationScreens 상태 업데이트
                            WidgetsBinding.instance.addPostFrameCallback((_) {
                              final navigationScreensState =
                                  context.findAncestorStateOfType<
                                      NavigationScreensState>();
                              if (navigationScreensState != null &&
                                  navigationScreensState.mounted) {
                                navigationScreensState.setState(() {
                                  navigationScreensState.selectedIndex = 2;
                                });
                              }
                            });

                            // 현재 페이지 스택 제거
                            if (Navigator.canPop(context)) {
                              Navigator.of(context).pop();
                            }
                          },
                        ),

                        const Spacer(), // 가운데 띄우기

                        // 화면 중앙에 타이틀 박스
                        Container(
                          height: 50,
                          padding: const EdgeInsets.symmetric(horizontal: 100),
                          decoration: BoxDecoration(
                            color: const Color(0xFFc06656),
                            borderRadius: BorderRadius.circular(30),
                          ),
                          alignment: Alignment.center,
                          child: Text(
                            widget.title, // 이전 페이지에서 전달된 title 사용
                            style: const TextStyle(
                              color: Colors.white,
                              fontWeight: FontWeight.bold,
                              fontSize: 25,
                            ),
                          ),
                        ),

                        const Spacer(), // 오른쪽도 균형 맞추기
                      ],
                    ),

                    // 오른쪽 상단: 재생 속도 조정 버튼
                    Positioned(
                      right: 0,
                      child: MenuAnchor(
                        style: const MenuStyle(
                          padding: WidgetStatePropertyAll(EdgeInsets.zero),
                          backgroundColor:
                              WidgetStatePropertyAll(Colors.transparent),
                        ),
                        menuChildren: [
                          Container(
                            margin: const EdgeInsets.only(top: 8),
                            padding: const EdgeInsets.symmetric(
                                horizontal: 4, vertical: 4),
                            decoration: BoxDecoration(
                              color: Colors.white,
                              borderRadius: BorderRadius.circular(8),
                              boxShadow: [
                                BoxShadow(
                                  color: Colors.black.withValues(alpha: 0.1),
                                  blurRadius: 4,
                                  offset: const Offset(0, 2),
                                ),
                              ],
                            ),
                            child: Row(
                              mainAxisSize: MainAxisSize.min,
                              children: [
                                _buildSpeedButton(
                                    '0.5x', _currentSpeed == '0.5x'),
                                _buildSpeedButton('1x', _currentSpeed == '1x'),
                                _buildSpeedButton(
                                    '1.5x', _currentSpeed == '1.5x'),
                                _buildSpeedButton('2x', _currentSpeed == '2x'),
                              ],
                            ),
                          ),
                        ],
                        builder: (context, controller, child) {
                          return GestureDetector(
                            onTap: () {
                              if (controller.isOpen) {
                                controller.close();
                              } else {
                                controller.open();
                              }
                            },
                            child: Container(
                              padding: const EdgeInsets.symmetric(
                                  horizontal: 12, vertical: 6),
                              decoration: BoxDecoration(
                                color: Colors.white,
                                borderRadius: BorderRadius.circular(8),
                                boxShadow: [
                                  BoxShadow(
                                    color: Colors.black.withValues(alpha: 0.1),
                                    blurRadius: 4,
                                    offset: const Offset(0, 2),
                                  ),
                                ],
                              ),
                              child: Text(
                                _currentSpeed,
                                style: const TextStyle(
                                  color: Color(0xFFE5958B),
                                  fontWeight: FontWeight.w500,
                                  fontSize: 14,
                                ),
                              ),
                            ),
                          );
                        },
                      ),
                    ),
                  ]),
                ),

                // 본문 영역 (악보 + 온셋 수 표시)
                Expanded(
                  child: Container(
                    margin: const EdgeInsets.all(16),
                    decoration: BoxDecoration(
                      color: Colors.white,
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: Stack(
                      children: [
                        // 악보 띄우기
                        Center(
                          child: Image.asset(
                            'assets/test/tom_mix.png',
                            fit: BoxFit.contain,
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
                ),

                // 하단 영역 (재생/녹음 버튼 + 녹음 상태 표시)
                Container(
                  margin: const EdgeInsets.fromLTRB(16, 0, 16, 16),
                  child: Center(
                    child: Container(
                      width: 450,
                      padding: const EdgeInsets.all(16),
                      decoration: BoxDecoration(
                        color: Colors.black.withValues(alpha: 0.03),
                        borderRadius: BorderRadius.circular(24),
                      ),
                      child: Column(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          // 오디오 재생 위치 표시하는 슬라이더
                          Container(
                            height: 2,
                            decoration: BoxDecoration(
                              color: Colors.white,
                              borderRadius: BorderRadius.circular(1),
                            ),
                            child: SliderTheme(
                              data: SliderTheme.of(context).copyWith(
                                activeTrackColor: const Color(0xFFE5958B),
                                inactiveTrackColor: Colors.transparent,
                                thumbColor: const Color(0xFFE5958B),
                                trackHeight: 2.0,
                                thumbShape: const RoundSliderThumbShape(
                                  enabledThumbRadius: 4.0,
                                ),
                                overlayShape: const RoundSliderOverlayShape(
                                  overlayRadius: 8.0,
                                ),
                              ),
                              child: Slider(
                                value: _currentPosition.clamp(
                                    0, _totalDuration > 0 ? _totalDuration : 1),
                                min: 0,
                                max: _totalDuration > 0 ? _totalDuration : 1,
                                onChanged: _seekAudio,
                              ),
                            ),
                          ),
                          const SizedBox(height: 16),

                          // 재생/녹음 토글 버튼
                          Row(
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: [
                              Container(
                                width: 48,
                                height: 48,
                                decoration: BoxDecoration(
                                  shape: BoxShape.circle,
                                  color: const Color(0xFFE5958B),
                                  boxShadow: [
                                    BoxShadow(
                                      color:
                                          Colors.black.withValues(alpha: 0.1),
                                      blurRadius: 4,
                                      offset: const Offset(0, 2),
                                    ),
                                  ],
                                ),
                                child: Material(
                                  color: Colors.transparent,
                                  child: InkWell(
                                    borderRadius: BorderRadius.circular(24),
                                    onTap: () {
                                      if (_isRecording) {
                                        _stopRecording();
                                      } else {
                                        _startAudio();
                                      }
                                    },
                                    child: Icon(
                                      _isRecording
                                          ? Icons.stop
                                          : Icons.play_arrow,
                                      color: Colors.white,
                                      size: 24,
                                    ),
                                  ),
                                ),
                              ),
                            ],
                          ),

                          // 현재 녹음 상태 표시
                          if (_recordingStatusMessage.isNotEmpty) ...[
                            const SizedBox(height: 16),
                            Text(
                              _recordingStatusMessage,
                              style: const TextStyle(
                                color: Color(0xFFE5958B),
                                fontSize: 14,
                                fontWeight: FontWeight.bold,
                              ),
                            ),
                          ],
                        ],
                      ),
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

          // 오버레이 - 카운트다운 숫자 표시
          if (isCountingDown)
            FadeTransition(
              opacity: _overlayAnimation,
              child: Container(
                color: Colors.black.withValues(alpha: 0.9),
                alignment: Alignment.center,
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    _buildCountdownNumber(3),
                    const SizedBox(width: 150),
                    _buildCountdownNumber(2),
                    const SizedBox(width: 150),
                    _buildCountdownNumber(1),
                  ],
                ),
              ),
            ),
        ],
      ),
    );
  }
}
