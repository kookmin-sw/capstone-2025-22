import 'package:capstone_2025/providers/metronome_provider.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:just_audio/just_audio.dart';
import 'package:capstone_2025/screens/mainPages/navigation_screens.dart';
import 'dart:async';

class MetronomeHeader extends ConsumerStatefulWidget {
  final AudioPlayer firstBeatPlayer;
  final AudioPlayer otherBeatPlayer;

  const MetronomeHeader({
    super.key,
    required this.firstBeatPlayer,
    required this.otherBeatPlayer,
  });

  @override
  ConsumerState<MetronomeHeader> createState() => _MetronomeHeaderState();
}

class _MetronomeHeaderState extends ConsumerState<MetronomeHeader> {
  Timer? _timer; // 메트로놈 소리 출력을 위한 타이머
  int next = 0;
  int subNext = 0;

  @override
  void initState() {
    // 위젯이 생성될 때 호출
    super.initState();
    preloadSounds();
  }

  Future<void> preloadSounds() async {
    // 사운드 미리 로드
    await widget.firstBeatPlayer.setAudioSource(
      AudioSource.asset('assets/sounds/metronome_first.wav'),
    );
    await widget.otherBeatPlayer.setAudioSource(
      AudioSource.asset('assets/sounds/metronome_other.wav'),
    );
  }

  void metronomeControl() {
    final isPlaying = ref.read(isPlayingProvider);
    if (isPlaying && _timer == null) return; // 상태 불일치 방지
    if (!isPlaying && _timer != null) return;

    if (_timer == null) {
      // 재생 시작
      try {
        widget.firstBeatPlayer.stop();
        widget.otherBeatPlayer.stop();
      } catch (e) {
        debugPrint('Audio stop error: $e');
      }

      ref.read(currItemProvider.notifier).state = 0;
      if (ref.watch(isSoundOnProvider)) {
        widget.firstBeatPlayer.seek(Duration.zero).then((_) {
          widget.firstBeatPlayer.play();
        });
      }

      _startMetronomeLoop();
      ref.read(isPlayingProvider.notifier).state = true;
    } else {
      // 정지
      stopMetronome();
    }
  }

  void updateMetronomeTiming() {
    if (_timer != null) {
      _timer?.cancel();
      _startMetronomeLoop(); // 인덱스 유지, 타이머만 새로 설정
    }
  }

  void _startMetronomeLoop() {
    int curBPM = ref.read(currBPMProvider).toInt();
    final timeSignature = ref.read(selectedTimeSignatureProvider);
    final currItemBase4 = ref.watch(currItemBase4Provider);
    String currBeatPattern = ref.watch(currBeatPatternProvider);
    int subBeatCount = 1;
    if (timeSignature == '1/4') {
      // 깜빡임 효과 위해 2개로 나누기
      curBPM = curBPM * 2;
      subBeatCount = 2;
    }

    if (currItemBase4) {
      switch (currBeatPattern) {
        case 'quarter':
          curBPM = curBPM * 1;
          break;
        case 'two':
          curBPM = curBPM * 2;
          subBeatCount = subBeatCount * 2;
          break;
        case 'triplet':
        case 'triplet2':
          curBPM = curBPM * 3;
          subBeatCount = subBeatCount * 3;
          break;
        case 'four':
        case 'four_2':
          curBPM = curBPM * 4;
          subBeatCount = subBeatCount * 4;
          break;
      }
    }

    _timer = Timer.periodic(Duration(milliseconds: 60000 ~/ curBPM), (timer) {
      final current = ref.read(currItemProvider);
      final timeSignature = ref.read(selectedTimeSignatureProvider);
      final beatCount = int.tryParse(timeSignature.split('/').first) ?? 4;
      final currItemBase4 = ref.watch(currItemBase4Provider);

      if (timeSignature == '1/4') {
        if (subNext % subBeatCount == 0) {
          next = (current == 0) ? -1 : 0;
          ref.read(currItemProvider.notifier).state = next;
        }
      } else {
        if (subNext % subBeatCount == 0) {
          next = (current + 1) % beatCount;
          ref.read(currItemProvider.notifier).state = next;
        }
      }

      final effectiveIndex = subNext;

      subNext = (subNext + 1) % (beatCount * subBeatCount);

      if (ref.watch(isSoundOnProvider)) {
        // 소리 재생을 위한 for loop
        if (currItemBase4) {
          if (effectiveIndex == 0) {
            widget.firstBeatPlayer.seek(Duration.zero);
            widget.firstBeatPlayer.play();
          } else {
            if (timeSignature == '1/4') {
              // 4분음표 기준일 때
              if ((currBeatPattern == 'quarter' ||
                      currBeatPattern == 'two' ||
                      currBeatPattern == 'triplet' ||
                      currBeatPattern == 'four') &&
                  effectiveIndex % subBeatCount == 0) {
                widget.firstBeatPlayer.seek(Duration.zero);
                widget.firstBeatPlayer.play();
              } else {}
            } else if (currBeatPattern == 'quarter' ||
                currBeatPattern == 'two' ||
                currBeatPattern == 'triplet' ||
                currBeatPattern == 'four') {
              widget.firstBeatPlayer.seek(Duration.zero);
              widget.firstBeatPlayer.play();
            } else if (currBeatPattern == 'triplet2') {
              if (effectiveIndex % 3 != 1) {
                widget.firstBeatPlayer.seek(Duration.zero);
                widget.firstBeatPlayer.play();
              } else {}
            } else if (currBeatPattern == 'four_2') {
              if (effectiveIndex % 4 == 0 || effectiveIndex % 4 == 3) {
                widget.firstBeatPlayer.seek(Duration.zero);
                widget.firstBeatPlayer.play();
              } else {}
            }
          }
        } else {
          // 8분음표 기준일 때
          if (next == 0) {
            widget.firstBeatPlayer.seek(Duration.zero);
            widget.firstBeatPlayer.play();
          } else {
            if (currBeatPattern == 'dot_quarter') {
              if (next % 3 == 0) {
                widget.otherBeatPlayer.seek(Duration.zero);
                widget.otherBeatPlayer.play();
              }
            } else if (currBeatPattern == 'three') {
              widget.otherBeatPlayer.seek(Duration.zero);
              widget.otherBeatPlayer.play();
            } else if (currBeatPattern == 'three_2') {
              if (next % 3 == 0 || next % 3 == 2) {
                widget.otherBeatPlayer.seek(Duration.zero);
                widget.otherBeatPlayer.play();
              } else {}
            }
          }
        }
      } else {
        // 소리 재생 여부가 false일 때
      }
    });
  }

  void stopMetronome() {
    _timer?.cancel();
    _timer = null;
    ref.read(isPlayingProvider.notifier).state = false;
    ref.read(currItemProvider.notifier).state = -1;
  }

  void clickedBackBtn() {
    // 뒤로가기 버튼 클릭 시 실행될 함수
    if (_timer != null) {
      // 타이머가 null이 아닐 때만 취소
      stopMetronome();
    }
    ref.read(currBPMProvider.notifier).state = 120; // BPM 초기화
    ref.read(currItemBase4Provider.notifier).state = true; // 4분음표 기준 초기화
    ref.read(currBeatPatternProvider.notifier).state = 'quater'; // 박자 패턴 초기화
    ref.read(selectedTimeSignatureProvider.notifier).state = '1/4'; // 박자 초기화
    ref.read(isSoundOnProvider.notifier).state = true; // 소리 재생 여부 초기화
    Navigator.pushReplacement(
      context,
      MaterialPageRoute(
          builder: (context) => NavigationScreens(firstSelectedIndex: 4)),
    );
  }

  @override
  void dispose() {
    _timer?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final isSoundOn = ref.watch(isSoundOnProvider);
    return Padding(
      padding: const EdgeInsets.only(top: 25),
      child: Row(
        children: [
          Expanded(
            // 뒤로가기 버튼
            flex: 1,
            child: IconButton(
              onPressed: clickedBackBtn,
              icon: const Icon(
                Icons.arrow_back_ios,
                size: 35,
              ),
            ),
          ),
          const Spacer(flex: 4),
          Align(
            // 메트로놈 재생 버튼
            alignment: Alignment.center,
            child: Container(
              // 재생 버튼 배경
              height: 50,
              width: 150,
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(30),
                boxShadow: [
                  BoxShadow(
                    color: Colors.black.withOpacity(0.1),
                    blurRadius: 10,
                    spreadRadius: 1,
                    offset: const Offset(0, 4),
                  ),
                ],
              ),
              child: Transform.translate(
                offset: const Offset(0, -12),
                child: IconButton(
                  onPressed: metronomeControl,
                  icon: Icon(
                    ref.watch(isPlayingProvider) // 재생 상태에 따라 아이콘 변경
                        ? Icons.stop
                        : Icons.play_arrow,
                    size: 60,
                  ),
                  highlightColor: Colors.transparent,
                  color: const Color(0xffD97D6C),
                ),
              ),
            ),
          ),
          const Spacer(flex: 4),
          Expanded(
            // 소리 설정 버튼
            flex: 1,
            child: IconButton(
              onPressed: () =>
                  {ref.read(isSoundOnProvider.notifier).state = !isSoundOn},
              icon: Icon(
                isSoundOn ? Icons.volume_up : Icons.volume_off,
                size: 45,
              ),
            ),
          ),
        ],
      ),
    );
  }
}
