import 'dart:io';
import 'dart:async';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter/foundation.dart';
import 'package:audioplayers/audioplayers.dart' as ap;
import 'package:flutter_sound/flutter_sound.dart' as fs;
import 'package:web_socket_channel/web_socket_channel.dart';
import 'package:capstone_2025/screens/mainPages/navigation_screens.dart';

class PatternFillScreen extends StatelessWidget {
  const PatternFillScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      theme: ThemeData(scaffoldBackgroundColor: const Color(0xFFF2F1F3)),
      home: const CountdownPage(),
      debugShowCheckedModeBanner: false,
    );
  }
}

class CountdownPage extends StatefulWidget {
  const CountdownPage({super.key});

  @override
  State<CountdownPage> createState() => _CountdownPageState();
}

class _CountdownPageState extends State<CountdownPage>
    with SingleTickerProviderStateMixin {
  int countdown = 3;
  bool isCountingDown = false;
  bool _isPlaying = false;
  bool _isRecording = false;
  final double _currentPosition = 0.0;
  final double _totalDuration = 0.0;
  String _currentSpeed = '1x'; // 현재 속도를 저장할 변수 추가

  late ap.AudioPlayer _audioPlayer;
  late fs.FlutterSoundRecorder _recorder;
  late AnimationController _overlayController;
  late Animation<double> _overlayAnimation;

  Timer? _countdownTimer;

  // WebSocket 관련 변수 주석 처리
  // late WebSocketChannel _channel;

  @override
  void initState() {
    super.initState();

    // audioPlayer 초기화
    _audioPlayer = ap.AudioPlayer();

    _recorder = fs.FlutterSoundRecorder();
    // WebSocket 연결 주석 처리
    // _channel = WebSocketChannel.connect(
    //     Uri.parse('ws://10.0.2.2:28080')); // 서버 URL 수정하기

    // 녹음기 초기화
    _recorder.openRecorder();

    // playerState 업데이트
    _audioPlayer.onPlayerStateChanged.listen((state) {
      if (state == ap.PlayerState.playing) {
        setState(() {
          _isPlaying = true;
        });
      } else {
        setState(() {
          _isPlaying = false;
        });
      }
    });

    // 애니메이션 컨트롤러 초기화
    _overlayController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 500),
    );
    _overlayAnimation =
        Tween<double>(begin: 0.0, end: 0.5).animate(_overlayController);
  }

  @override
  void dispose() {
    _countdownTimer?.cancel(); // 타이머 취소
    _recorder.closeRecorder(); // Recorder dispose 처리
    _audioPlayer.dispose(); // AudioPlayer dispose 처리
    _overlayController.dispose(); // AnimationController dispose 처리
    // WebSocket 종료 주석 처리
    // _channel.sink.close();
    super.dispose();
  }

  // 시범 연주를 시작하는 함수
  void _startAudio() async {
    // WAV 파일 재생
    await _audioPlayer.play(ap.AssetSource('test/tom_mix.wav'));

    // WAV 파일 재생이 끝날 때까지 대기
    _audioPlayer.onPlayerComplete.listen((event) {
      _startCountdown(); // WAV 파일 재생이 끝나면 카운트다운 시작
    });
  }

  void _pauseAudio() async {
    await _audioPlayer.pause();
    setState(() {
      _isPlaying = false;
    });
  }

  void _seekAudio(double position) async {
    await _audioPlayer.seek(Duration(seconds: position.toInt()));
  }

  // 녹음 시작 함수
  void _startRecording() async {
    if (_isRecording) return; // 이미 녹음 중이면 함수 종료

    await _recorder.startRecorder(
        toFile: 'drum_performance.aac'); // 드럼 연주 녹음 시작
    setState(() {
      _isRecording = true;
    });
  }

  // 카운트다운을 시작하는 함수
  void _startCountdown() {
    setState(() {
      isCountingDown = true;
      countdown = 3;
    });

    _overlayController.forward(); // 카운트다운 애니메이션 시작

    _countdownTimer = Timer.periodic(const Duration(seconds: 1), (timer) {
      if (countdown == 0) {
        timer.cancel();
        _overlayController.reverse().then((_) {
          if (mounted) {
            setState(() {
              isCountingDown = false;
            });
            // 카운트다운 후 사용자 연주 시작
            _startRecording(); // 사용자 연주 녹음 시작
          }
        });
      } else {
        if (mounted) {
          setState(() {
            countdown--;
          });
        }
      }
    });
  }

  // 녹음 종료 함수
  void _stopRecording() async {
    String? filePath = await _recorder.stopRecorder(); // 녹음 종료 후 파일 경로 반환
    setState(() {
      _isRecording = false;
    });

    // 녹음된 오디오 파일을 서버로 전송
    if (filePath != null) {
      File file = File(filePath);
      List<int> bytes = await file.readAsBytes();

      // 서버로 오디오 파일 전송
      _sendAudioToServer(bytes);
    }
  }

  // 녹음된 오디오 파일을 WebSocket을 통해 서버로 전송
  void _sendAudioToServer(List<int> bytes) {
    // WebSocket 전송 주석 처리
    // _channel.sink.add(Uint8List.fromList(bytes));
    print('서버 연결이 비활성화되어 있습니다.');
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
              color: Colors.black.withOpacity(0.5),
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
          setState(() {
            _audioPlayer.setPlaybackRate(newSpeed);
            _currentSpeed = speed; // 현재 속도 업데이트
          });
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
                  child: Row(
                    children: [
                      IconButton(
                        icon: const Icon(Icons.home_filled),
                        onPressed: () {
                          Navigator.push(
                            context,
                            MaterialPageRoute(
                                builder: (_) => const NavigationScreens()),
                          );
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
                        child: const Text(
                          'Basic Pattern 1',
                          style: TextStyle(
                            color: Colors.white,
                            fontWeight: FontWeight.bold,
                            fontSize: 25,
                          ),
                        ),
                      ),
                      const Spacer(),
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
                                  _buildSpeedButton('0.5x', false),
                                  _buildSpeedButton('1x', true),
                                  _buildSpeedButton('1.5x', false),
                                  _buildSpeedButton('2x', false),
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
                                      color:
                                          Colors.black.withValues(alpha: 0.1),
                                      blurRadius: 4,
                                      offset: const Offset(0, 2),
                                    ),
                                  ],
                                ),
                                child: Text(
                                  _currentSpeed, // 현재 속도 표시
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
                    ],
                  ),
                ),
                Expanded(
                  child: Container(
                    margin: const EdgeInsets.all(16),
                    decoration: BoxDecoration(
                      color: Colors.white,
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: Center(
                      child: Image.asset(
                        'assets/test/tom_mix.png',
                        fit: BoxFit.contain,
                      ),
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
                                value: _currentPosition,
                                min: 0,
                                max: _totalDuration,
                                onChanged: _seekAudio,
                              ),
                            ),
                          ),
                          const SizedBox(height: 16),
                          Container(
                            width: 48,
                            height: 48,
                            decoration: BoxDecoration(
                              shape: BoxShape.circle,
                              color: const Color(0xFFE5958B),
                              boxShadow: [
                                BoxShadow(
                                  color: Colors.black.withValues(alpha: 0.1),
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
                                  if (_isPlaying) {
                                    _pauseAudio();
                                  } else {
                                    _startAudio();
                                  }
                                },
                                child: Icon(
                                  _isPlaying ? Icons.pause : Icons.play_arrow,
                                  color: Colors.white,
                                  size: 24,
                                ),
                              ),
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                ),
                // 녹음 상태 표시
                if (_isRecording) ...[
                  // 녹음 종료 버튼 삭제
                ],
              ],
            ),
          ),
          if (isCountingDown)
            FadeTransition(
              opacity: _overlayAnimation,
              child: Container(
                color: Colors.black.withValues(alpha: 0.9), // 어두운 반투명 오버레이
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
