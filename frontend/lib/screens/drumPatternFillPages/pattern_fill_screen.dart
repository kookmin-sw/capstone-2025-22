import 'dart:async';
import 'package:flutter/material.dart';
// ignore: depend_on_referenced_packages
import 'package:path_provider/path_provider.dart';
import 'package:audioplayers/audioplayers.dart' as ap;
import 'package:capstone_2025/widgets/drum_recording_widget.dart';
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
  bool _isPlaying = false;
  bool _showPracticeMessage = false;
  double _currentPosition = 0.0;
  double _totalDuration = 0.0;
  String _currentSpeed = '1x';
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

  String? _recordingPath; // 녹음 파일 경로 추가

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

    // 녹음 경로 초기화 (덮어쓰기 방식)
    _initializeRecording();
  }

  void _initializeRecording() async {
    final appDocDir = await getApplicationDocumentsDirectory();
    _recordingPath = '${appDocDir.path}/current_performance.wav'; // 녹음 파일 경로
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

  // 시범 연주를 재생하는 함수
  Future<void> _startAudio() async {
    if (!mounted) return;

    // 시범 연주 시작 메시지 표시
    setState(() => _showPracticeMessage = true);
    _overlayController.forward();

    // 1초 후 메시지 숨기기
    await Future.delayed(const Duration(milliseconds: 1000));

    if (!mounted) return;
    _overlayController.reverse().then((_) {
      if (mounted) setState(() => _showPracticeMessage = false);
    });

    // 시범 연주 오디오 재생
    await _audioPlayer.play(ap.AssetSource('test/tom_mix.wav'));

    // 시범 연주의 총 길이를 저장 및 출력
    _audioPlayer.onDurationChanged.listen((duration) {
      if (mounted) {
        _totalDuration = duration.inSeconds.toDouble();
        print('🎵 시범 연주 총 길이: ${_totalDuration.toStringAsFixed(2)}초');
      }
    });

    // 시범 연주가 끝나면 카운트다운 시작
    _playerCompleteSubscription = _audioPlayer.onPlayerComplete.listen((event) {
      if (mounted) {
        print('⏱️ 시범 연주 종료! (총 길이: ${_totalDuration.toStringAsFixed(2)}초)');

        // DrumRecordingWidget의 카운트다운 시작
        final drumRecordingState = _drumRecordingKey.currentState;
        if (drumRecordingState != null) {
          drumRecordingState.startCountdown(
            onCountdownComplete: () {
              drumRecordingState.startRecording();
            },
          );
        }
      }
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

  // 페이지가 종료될 때 리소스 해제하는 함수
  @override
  void dispose() {
    _practiceMessageTimer?.cancel();
    _positionUpdateTimer?.cancel();
    _playerStateSubscription?.cancel();
    _playerCompleteSubscription?.cancel();
    _positionSubscription?.cancel();

    _audioPlayer.dispose(); // 오디오플레이어 정리
    _overlayController.dispose(); // 애니메이션 컨트롤러

    super.dispose();
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

                            // DrumRecordingWidget의 녹음 중지
                            final drumRecordingState =
                                _drumRecordingKey.currentState;
                            if (drumRecordingState != null &&
                                drumRecordingState.isRecording) {
                              drumRecordingState.stopRecording();
                            }

                            // 리소스 해제
                            _playerStateSubscription?.cancel();
                            _playerCompleteSubscription?.cancel();
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
                                      final drumRecordingState =
                                          _drumRecordingKey.currentState;
                                      if (drumRecordingState != null &&
                                          drumRecordingState.isRecording) {
                                        drumRecordingState.stopRecording();
                                      } else {
                                        _startAudio();
                                      }
                                    },
                                    child: Icon(
                                      _drumRecordingKey
                                                  .currentState?.isRecording ??
                                              false
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
                          if (_drumRecordingKey.currentState
                                  ?.recordingStatusMessage.isNotEmpty ??
                              false) ...[
                            const SizedBox(height: 16),
                            Text(
                              _drumRecordingKey
                                  .currentState!.recordingStatusMessage,
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

          // DrumRecordingWidget 추가 (보이지 않지만 기능 사용)
          Offstage(
            offstage: true, // UI를 화면에 표시하지 않음
            child: DrumRecordingWidget(
              key: _drumRecordingKey,
              title: widget.title,
              xmlFilePath: 'assets/test/tom_mix.xml',
              audioFilePath: 'assets/test/tom_mix.wav',
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
