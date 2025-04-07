import 'package:audioplayers/audioplayers.dart';
import 'package:capstone_2025/screens/mainPages/navigation_screens.dart';
import 'package:flutter/material.dart';
import 'dart:async';
import 'widgets/metronome_header.dart';
import 'widgets/metronome_beats.dart';
import 'widgets/metronome_controls.dart';

class Metronome extends StatefulWidget {
  const Metronome({super.key});

  @override
  State<Metronome> createState() => _MetronomeState();
}

class _MetronomeState extends State<Metronome> {
  final AudioPlayer firstBeatPlayer = AudioPlayer();
  final AudioPlayer otherBeatPlayer = AudioPlayer();

  bool isHitVisible = true;
  bool isSoundOn = true;
  int currItem = -1;
  bool currItemBase4 = true;
  Timer? _timer;
  String selectedTimeSignature = '1/4';
  int curBPM = 120;
  final TextEditingController bpmController = TextEditingController();
  final FocusNode bpmFocusNode = FocusNode();

  @override
  void initState() {
    super.initState();
    bpmFocusNode.addListener(() {
      setState(() {});
      if (bpmFocusNode.hasFocus) {
        bpmController.clear();
      } else {
        final bpm = int.tryParse(bpmController.text);
        if (bpm != null && bpm >= 10 && bpm <= 400) {
          setState(() {
            curBPM = bpm;
          });
          bpmController.text = curBPM.toString();
        } else {
          bpmController.text = curBPM.toString();
        }
      }
    });
    preloadSounds();
    bpmController.text = curBPM.toString();
  }

  Future<void> preloadSounds() async {
    await firstBeatPlayer.setReleaseMode(ReleaseMode.stop);
    await otherBeatPlayer.setReleaseMode(ReleaseMode.stop);
    await firstBeatPlayer.setSource(AssetSource('sounds/metronome_first.wav'));
    await otherBeatPlayer.setSource(AssetSource('sounds/metronome_other.wav'));
  }

  @override
  void dispose() {
    _timer?.cancel();
    bpmController.dispose();
    bpmFocusNode.dispose();
    super.dispose();
  }

  void clickedBackBtn() {
    if (_timer != null) {
      _timer?.cancel();
      _timer = null;
    }
    Navigator.pushReplacement(
      context,
      MaterialPageRoute(
          builder: (context) => NavigationScreens(firstSelectedIndex: 4)),
    );
  }

  void startMetronome() {
    if (_timer == null) {
      setState(() {
        currItem = 0;
      });

      if (isSoundOn) {
        if (currItemBase4) {
          // 4분음표 (1/4, 2/4, 3/4, 4/4)
          firstBeatPlayer.seek(Duration.zero);
          firstBeatPlayer.resume();
        } else {
          // 8분음표 (3/8, 6/8, 9/8, 12/8)
          if (currItem % 3 == 0) {
            firstBeatPlayer.seek(Duration.zero);
            firstBeatPlayer.resume();
          } else {
            otherBeatPlayer.seek(Duration.zero);
            otherBeatPlayer.resume();
          }
        }
      }

      _timer = Timer.periodic(
        Duration(milliseconds: 60000 ~/ curBPM),
        (timer) {
          setState(() {
            currItem =
                (currItem + 1) % int.parse(selectedTimeSignature.split('/')[0]);
          });

          if (isSoundOn) {
            if (currItemBase4) {
              // 4분음표 - 첫 박자에 first 소리, 나머지는 other 소리 재생
              if (currItem == 0) {
                firstBeatPlayer.seek(Duration.zero);
                firstBeatPlayer.resume();
              } else {
                otherBeatPlayer.seek(Duration.zero);
                otherBeatPlayer.resume();
              }
            } else {
              // 8분음표 - 3박자마다 first 소리, 나머지는 other 소리 재생
              if (currItem % 3 == 0) {
                firstBeatPlayer.seek(Duration.zero);
                firstBeatPlayer.resume();
              } else {
                otherBeatPlayer.seek(Duration.zero);
                otherBeatPlayer.resume();
              }
            }
          }
        },
      );
    } else {
      _timer?.cancel();
      _timer = null;
      setState(() {
        currItem = -1;
      });
    }
  }

  void changeBeat(String beat) {
    setState(() {
      selectedTimeSignature = beat;
      if (beat == '3/8' || beat == '6/8' || beat == '9/8' || beat == '12/8') {
        currItemBase4 = false;
      } else {
        currItemBase4 = true;
      }
    });
  }

  void beatChanged() {
    // 비트 패턴 변경
    return;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: GestureDetector(
        onTap: () {
          FocusScope.of(context).unfocus();
        },
        behavior: HitTestBehavior.translucent,
        child: Column(
          children: [
            Expanded(
              flex: 2,
              child: MetronomeHeader(
                onBackPressed: clickedBackBtn,
                onPlayPressed: startMetronome,
                isPlaying: _timer != null,
                isHitVisible: isHitVisible,
                isSoundOn: isSoundOn,
                onHitVisibleChanged: (val) {
                  setState(() {
                    isHitVisible = val;
                  });
                  WidgetsBinding.instance.addPostFrameCallback((_) {
                    setState(() {});
                  });
                },
                onSoundOnChanged: (val) {
                  setState(() {
                    isSoundOn = val;
                  });
                  WidgetsBinding.instance.addPostFrameCallback((_) {
                    setState(() {});
                  });
                },
              ),
            ),
            const SizedBox(height: 20),
            Expanded(
              flex: selectedTimeSignature == '9/8' ||
                      selectedTimeSignature == '12/8'
                  ? 4
                  : 3,
              child: MetronomeBeats(
                currentBeat: currItem,
                timeSignature: selectedTimeSignature,
                currItemBase4: currItemBase4,
                isSoundOn: isSoundOn,
                isPlaying: _timer != null,
              ),
            ),
            Expanded(
              flex: 4,
              child: MetronomeControls(
                selectedTimeSignature: selectedTimeSignature,
                curBPM: curBPM,
                bpmController: bpmController,
                bpmFocusNode: bpmFocusNode,
                onTimeSignatureChanged: changeBeat,
                onBPMChanged: (bpm) {
                  bool wasPlaying = _timer != null;

                  if (wasPlaying) {
                    _timer?.cancel();
                    _timer = null;
                  }

                  setState(() {
                    curBPM = bpm;
                    bpmController.text = curBPM.toString();
                    currItem = -1;
                  });

                  if (wasPlaying) {
                    startMetronome();
                  }
                },
                onBeatChanged: beatChanged,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
