import 'dart:io';
import 'dart:async';
import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';
import 'package:audioplayers/audioplayers.dart' as ap;
import 'package:flutter_sound/flutter_sound.dart' as fs;
import 'package:stomp_dart_client/stomp_dart_client.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import 'package:capstone_2025/screens/mainPages/navigation_screens.dart';

class PatternFillScreen extends StatelessWidget {
  final String title;

  const PatternFillScreen({super.key, required this.title});

  @override
  Widget build(BuildContext context) {
    return CountdownPage(title: title);
  }
}

class CountdownPage extends StatefulWidget {
  final String title;

  const CountdownPage({super.key, required this.title});

  @override
  State<CountdownPage> createState() => _CountdownPageState();
}

class _CountdownPageState extends State<CountdownPage>
    with SingleTickerProviderStateMixin {
  int countdown = 3;
  bool isCountingDown = false;
  bool _isPlaying = false;
  bool _isRecording = false;
  double _currentPosition = 0.0;
  double _totalDuration = 0.0;
  String _currentSpeed = '1x';
  bool _showPracticeMessage = false;
  String _recordingStatus = '';
  String _userEmail = '';
  List<dynamic> _detectedOnsets = [];

  late ap.AudioPlayer _audioPlayer;
  late fs.FlutterSoundRecorder _recorder;
  late AnimationController _overlayController;
  late Animation<double> _overlayAnimation;
  late StompClient _stompClient;
  final _storage = const FlutterSecureStorage();
  String? _recordingPath;

  Timer? _countdownTimer;
  Timer? _practiceMessageTimer;
  Timer? _positionUpdateTimer;
  Timer? _recordingDataTimer;

  StreamSubscription? _playerStateSubscription;
  StreamSubscription? _playerCompleteSubscription;
  StreamSubscription? _positionSubscription;
  StreamSubscription<fs.RecordingDisposition>? _recorderSubscription;

  @override
  void initState() {
    super.initState();
    _initializeData();
  }

  Future<void> _initializeData() async {
    _userEmail = await _storage.read(key: 'user_email') ?? 'test@example.com';

    _audioPlayer = ap.AudioPlayer();
    _recorder = fs.FlutterSoundRecorder();
    await _initRecorder();

    _overlayController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 500),
    );
    _overlayAnimation =
        Tween<double>(begin: 0.0, end: 1.0).animate(_overlayController);

    _setupAudioListeners();
    _setupWebSocket();
  }

  Future<void> _initRecorder() async {
    await _recorder.openRecorder();
    // Get application documents directory for storing recordings
    final appDocDir = await getApplicationDocumentsDirectory();
    _recordingPath = '${appDocDir.path}/drum_performance.wav';
  }

  void _setupAudioListeners() {
    _playerStateSubscription =
        _audioPlayer.onPlayerStateChanged.listen((state) {
      if (!mounted) return;
      setState(() {
        _isPlaying = state == ap.PlayerState.playing;
      });
    });

    _positionSubscription = _audioPlayer.onPositionChanged.listen((position) {
      if (!mounted) return;
      setState(() {
        _currentPosition = position.inSeconds.toDouble();
      });
    });

    _audioPlayer.onDurationChanged.listen((duration) {
      if (!mounted) return;
      setState(() {
        _totalDuration = duration.inSeconds.toDouble();
      });
    });
  }

  void _setupWebSocket() {
    _stompClient = StompClient(
      config: StompConfig(
        url: 'ws://10.0.2.2:28080/ws/audio',
        onConnect: (StompFrame frame) {
          print('WebSocket connected successfully!');
          _stompClient.subscribe(
            destination: '/topic/onset/$_userEmail',
            callback: (frame) {
              if (frame.body != null) {
                final response = json.decode(frame.body!);
                print('Received WebSocket data:');
                print('Full response: $response');

                if (response.containsKey('onsets')) {
                  setState(() {
                    _detectedOnsets = response['onsets'];
                  });
                  print('Onsets: ${response['onsets']}');
                }
                print('Timestamp: ${DateTime.now()}');
              } else {
                print('Received empty WebSocket frame');
              }
            },
          );
        },
        onWebSocketError: (dynamic error) {
          print('WebSocket error: $error');
          print('Error occurred at: ${DateTime.now()}');
        },
        onDisconnect: (frame) {
          print('WebSocket disconnected at: ${DateTime.now()}');
        },
      ),
    );

    // Activate WebSocket connection
    _stompClient.activate();
  }

  void _startAudio() async {
    if (!mounted) return;

    // 시범 연주 시작 메시지 표시
    setState(() {
      _showPracticeMessage = true;
    });
    _overlayController.forward();

    // 1초 후 메시지 숨기기 (0.5초에서 1초로 조정)
    await Future.delayed(const Duration(milliseconds: 1000));

    if (!mounted) return;

    _overlayController.reverse().then((_) {
      if (!mounted) return;
      setState(() {
        _showPracticeMessage = false;
      });
    });

    // 메시지가 사라진 후 바로 시범 연주 시작
    await _audioPlayer.play(ap.AssetSource('test/tom_mix.wav'));

    // 시범 연주가 끝나면 카운트다운 시작
    _playerCompleteSubscription = _audioPlayer.onPlayerComplete.listen((event) {
      if (!mounted) return;
      _startCountdown();
    });
  }

  void _startRecording() async {
    if (_isRecording || !mounted) return;

    try {
      print("Starting recording to $_recordingPath");
      await _recorder.startRecorder(
        toFile: _recordingPath,
        codec: fs.Codec.pcm16WAV,
      );

      setState(() {
        _isRecording = true;
        _recordingStatus = '녹음 시작됨';
      });

      // Set up timer to send recording data every second
      _recordingDataTimer = Timer.periodic(const Duration(seconds: 1), (timer) {
        _sendRecordingData();
      });
    } catch (e) {
      setState(() {
        _recordingStatus = '녹음 시작 실패: $e';
      });
      print('Recording error: $e');
    }
  }

  void _stopRecording() async {
    if (!_isRecording || !mounted) return;

    _recordingDataTimer?.cancel();
    await _recorder.stopRecorder();

    // Send final recording data
    _sendRecordingData();

    setState(() {
      _isRecording = false;
      _recordingStatus = '녹음 완료';
    });
  }

  void _sendRecordingData() async {
    if (!_stompClient.connected) {
      print('WebSocket is not connected!');
      return;
    }

    try {
      final file = File(_recordingPath!);
      if (await file.exists()) {
        final bytes = await file.readAsBytes();
        final base64String = base64Encode(bytes);

        final message = {
          'email': _userEmail,
          'message': base64String,
        };

        print('Sending WebSocket data at: ${DateTime.now()}');

        _stompClient.send(
          destination: '/app/audio/forwarding',
          body: json.encode(message),
          headers: {
            'content-type': 'application/json',
          },
        );

        setState(() {
          _recordingStatus = '녹음 중... 데이터 전송됨';
        });
      } else {
        print('Recording file not found: $_recordingPath');
      }
    } catch (e) {
      print('Error sending recording data: $e');
    }
  }

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

    if (_isRecording) {
      _recorder.stopRecorder();
    }

    _recorder.closeRecorder();
    _audioPlayer.dispose();
    _overlayController.dispose();

    if (_stompClient.connected) {
      _stompClient.deactivate();
    }

    super.dispose();
  }

  void _pauseAudio() async {
    if (!mounted) return;
    await _audioPlayer.pause();
    setState(() {
      _isPlaying = false;
    });
  }

  void _seekAudio(double position) async {
    if (!mounted) return;
    await _audioPlayer.seek(Duration(seconds: position.toInt()));
  }

  // 카운트다운을 시작하는 함수
  void _startCountdown() {
    if (!mounted) return;

    setState(() {
      isCountingDown = true;
      countdown = 3;
    });

    _overlayController.forward();

    _countdownTimer = Timer.periodic(const Duration(seconds: 1), (timer) {
      if (!mounted) {
        timer.cancel();
        return;
      }

      if (countdown == 0) {
        timer.cancel();
        _overlayController.reverse().then((_) async {
          if (!mounted) return;
          setState(() {
            isCountingDown = false;
          });

          // 카운트다운이 끝나면 바로 사용자 녹음 시작
          _startRecording();
        });
      } else {
        setState(() {
          countdown--;
        });
      }
    });
  }

  Widget buildNumber(int number) {
    final isHighlighted = number == countdown;

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

  Widget _buildSpeedButton(String speed, bool isSelected) {
    return Material(
      color: Colors.transparent,
      child: InkWell(
        onTap: () {
          double newSpeed = 1.0;
          switch (speed) {
            case '0.5x':
              newSpeed = 0.5;
              break;
            case '1x':
              newSpeed = 1.0;
              break;
            case '1.5x':
              newSpeed = 1.5;
              break;
            case '2x':
              newSpeed = 2.0;
              break;
          }
          if (mounted) {
            setState(() {
              _audioPlayer.setPlaybackRate(newSpeed);
              _currentSpeed = speed; // 현재 속도 업데이트
            });
          }
          Navigator.of(context).pop();
        },
        borderRadius: BorderRadius.circular(8),
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

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFF2F1F3),
      body: Stack(
        children: [
          SafeArea(
            child: Column(
              children: [
                const SizedBox(height: 10),
                Padding(
                  padding:
                      const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
                  child: Stack(children: [
                    Row(
                      children: [
                        IconButton(
                          icon: const Icon(Icons.home_filled),
                          onPressed: () {
                            // 오디오 재생 중지
                            if (_isPlaying) {
                              _audioPlayer.stop();
                            }

                            // 녹음 중지
                            if (_isRecording) {
                              _stopRecording();
                            }

                            // 리소스 정리
                            _playerStateSubscription?.cancel();
                            _playerCompleteSubscription?.cancel();
                            _countdownTimer?.cancel();
                            _practiceMessageTimer?.cancel();
                            _positionUpdateTimer?.cancel();

                            // NavigationScreens 상태 업데이트
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

                            // 네비게이션 처리
                            if (Navigator.canPop(context)) {
                              Navigator.of(context).pop();
                            }
                          },
                        ),
                        const Spacer(),
                        Container(
                          height: 50,
                          padding: const EdgeInsets.symmetric(horizontal: 100),
                          decoration: BoxDecoration(
                            color: const Color(0xFFc06656),
                            borderRadius: BorderRadius.circular(30),
                          ),
                          alignment: Alignment.center,
                          child: Text(
                            widget.title,
                            style: const TextStyle(
                              color: Colors.white,
                              fontWeight: FontWeight.bold,
                              fontSize: 25,
                            ),
                          ),
                        ),
                        const Spacer(),
                      ],
                    ),
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
                Expanded(
                  child: Container(
                    margin: const EdgeInsets.all(16),
                    decoration: BoxDecoration(
                      color: Colors.white,
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: Stack(
                      children: [
                        Center(
                          child: Image.asset(
                            'assets/test/tom_mix.png',
                            fit: BoxFit.contain,
                          ),
                        ),
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
                          if (_recordingStatus.isNotEmpty) ...[
                            const SizedBox(height: 16),
                            Text(
                              _recordingStatus,
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
          if (isCountingDown)
            FadeTransition(
              opacity: _overlayAnimation,
              child: Container(
                color: Colors.black.withValues(alpha: 0.9),
                alignment: Alignment.center,
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    buildNumber(3),
                    const SizedBox(width: 150),
                    buildNumber(2),
                    const SizedBox(width: 150),
                    buildNumber(1),
                  ],
                ),
              ),
            ),
        ],
      ),
    );
  }
}
