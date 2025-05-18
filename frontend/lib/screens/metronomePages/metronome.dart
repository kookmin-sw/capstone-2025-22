import 'package:just_audio/just_audio.dart';
import 'package:capstone_2025/screens/metronomePages/widgets/metronome_beats.dart';
import 'package:capstone_2025/screens/metronomePages/widgets/metronome_controls.dart';
import 'package:capstone_2025/screens/metronomePages/widgets/metronome_header.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:capstone_2025/providers/metronome_provider.dart';
import 'package:flutter/material.dart';
import 'package:flutter_screenutil/flutter_screenutil.dart';

class Metronome extends ConsumerStatefulWidget {
  const Metronome({super.key});

  @override
  ConsumerState<Metronome> createState() => _MetronomeState();
}

class _MetronomeState extends ConsumerState<Metronome> {
  // 오디오 플레이어
  final AudioPlayer firstBeatPlayer = AudioPlayer(); // 첫 박자
  final AudioPlayer otherBeatPlayer = AudioPlayer(); // 나머지 박자
  final GlobalKey metronomeHeaderKey = GlobalKey();

  @override
  Widget build(BuildContext context) {
    String selectedTimeSignature =
        ref.watch(selectedTimeSignatureProvider).toString();

    return Scaffold(
      resizeToAvoidBottomInset: true,
      body: GestureDetector(
        onTap: () => FocusScope.of(context).unfocus(),
        behavior: HitTestBehavior.translucent,
        child: Column(
          children: [
            Expanded(
              flex: (selectedTimeSignature == '9/8' ||
                      selectedTimeSignature == '12/8')
                  ? 3
                  : 2,
              child: MetronomeHeader(
                key: metronomeHeaderKey,
                firstBeatPlayer: firstBeatPlayer,
                otherBeatPlayer: otherBeatPlayer,
              ),
            ),
            SizedBox(
              height: (selectedTimeSignature == '9/8' ||
                      selectedTimeSignature == '12/8')
                  ? 5.h
                  : 30.h,
            ),
            Expanded(
              flex: (selectedTimeSignature == '9/8' ||
                      selectedTimeSignature == '12/8')
                  ? 7
                  : 5,
              child: MetronomeBeats(),
            ),
            Expanded(
              flex: (selectedTimeSignature == '9/8' ||
                      selectedTimeSignature == '12/8')
                  ? 5
                  : 4,
              child: MetronomeControls(
                metronomeHeaderKey: metronomeHeaderKey,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
