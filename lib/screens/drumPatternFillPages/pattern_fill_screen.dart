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
  double _currentPosition = 0.0;
  double _totalDuration = 0.0;

  late ap.AudioPlayer _audioPlayer;
  late fs.FlutterSoundRecorder _recorder;
  late AnimationController _overlayController;
  late Animation<double> _overlayAnimation;
  late WebSocketChannel _channel;

  @override
  void initState() {
    super.initState();

    // audioPlayer 초기화
    _audioPlayer = ap.AudioPlayer();

    _recorder = fs.FlutterSoundRecorder();
    _channel = WebSocketChannel.connect(
        Uri.parse('ws://10.0.2.2:28080')); // 서버 URL 수정하기

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
    _recorder.closeRecorder(); // Recorder dispose 처리
    _audioPlayer.dispose(); // AudioPlayer dispose 처리
    _overlayController.dispose(); // AnimationController dispose 처리
    _channel.sink.close(); // WebSocket 종료
    super.dispose();
  }

  // 시범 연주를 시작하는 함수
  void _startAudio() async {
    _audioPlayer.play(ap.AssetSource('assets/test/tom_mix.wav'));
    _audioPlayer.onDurationChanged.listen((duration) {
      setState(() {
        _totalDuration = duration.inSeconds.toDouble();
      });
    });

    _startCountdown(); // 시범 연주가 끝나면 바로 카운트다운 실행
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

  // 카운트다운을 시작하는 함수
  void _startCountdown() {
    setState(() {
      isCountingDown = true;
      countdown = 3;
    });

    _overlayController.forward(); // 카운트다운 애니메이션 시작

    Timer.periodic(const Duration(seconds: 1), (timer) {
      if (countdown == 0) {
        timer.cancel();
        _overlayController.reverse().then((_) {
          setState(() {
            isCountingDown = false;
          });
          // 카운트다운 후 사용자 연주 시작
          _startRecording(); // 사용자 연주 녹음 시작
        });
      } else {
        setState(() {
          countdown--;
        });
      }
    });
  }

  // 녹음 시작 함수
  void _startRecording() async {
    await _recorder.startRecorder(
        toFile: 'drum_performance.aac'); // 드럼 연주 녹음 시작
    setState(() {
      _isRecording = true;
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
    _channel.sink.add(Uint8List.fromList(bytes)); // 오디오 파일을 서버로 전송
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
                      DropdownButton<String>(
                        value: '1x', // 선택된 배속 값
                        underline: const SizedBox(),
                        items: ['0.5x', '1x', '1.5x', '2x'].map((speed) {
                          return DropdownMenuItem(
                            value: speed,
                            child: Text(
                              speed,
                              style: TextStyle(
                                fontWeight: FontWeight.bold,
                                color:
                                    speed == '1x' ? Colors.red : Colors.black87,
                              ),
                            ),
                          );
                        }).toList(),
                        onChanged: (value) {
                          // 박자 변경 시 처리 로직
                          double newSpeed = 1.0;
                          switch (value) {
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
                          _audioPlayer.setPlaybackRate(newSpeed); // 재생 속도 설정
                          setState(() {
                            _isPlaying = false;
                            _currentPosition = 0.0; // 박자 변경 시 슬라이더 초기화
                          });
                        },
                      ),
                    ],
                  ),
                ),
                Expanded(
                  child: Center(
                    child: Container(
                      // margin: const EdgeInsets.symmetric(horizontal: 16),
                      decoration: BoxDecoration(
                        color: Colors.white,
                        borderRadius: BorderRadius.circular(16),
                      ),
                      child: Center(
                        child: Image.asset(
                          'assets/test/tom_mix.png', // 악보 이미지
                          fit: BoxFit.contain,
                        ),
                      ),
                    ),
                  ),
                ),
                const SizedBox(height: 16),
                // 재생 컨트롤러
                // 기존의 "Slider" 위젯과 다른 UI 요소들 수정:
                Container(
                  margin: const EdgeInsets.only(bottom: 24),
                  padding:
                      const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
                  decoration: BoxDecoration(
                    color: Colors.grey.shade200,
                    borderRadius: BorderRadius.circular(24),
                  ),
                  child: Row(
                    children: [
                      IconButton(
                        icon: Icon(
                          _isPlaying ? Icons.pause : Icons.play_arrow,
                          color: Colors.brown,
                          size: 36,
                        ),
                        onPressed: () {
                          if (_isPlaying) {
                            _pauseAudio(); // 음성 정지 함수
                          } else {
                            _startAudio(); // 음성 시작 함수
                          }
                        },
                      ),
                      const SizedBox(width: 8),
                      Expanded(
                        child: Slider(
                          value: _currentPosition,
                          min: 0,
                          max: _totalDuration,
                          onChanged: (v) {
                            // 슬라이더 이동 시 음악 진행 상태 업데이트
                            _seekAudio(v);
                          },
                          activeColor: Color(0xffc06656),
                        ),
                      ),
                      const SizedBox(width: 8),
                    ],
                  ),
                ),
                // 녹음 상태 표시
                if (_isRecording) ...[
                  ElevatedButton(
                    onPressed: _stopRecording,
                    child: Text('녹음 종료'),
                  ),
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
