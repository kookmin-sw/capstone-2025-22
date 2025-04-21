import 'package:capstone_2025/providers/metronome_provider.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

class MetronomeBeats extends ConsumerStatefulWidget {
  @override
  ConsumerState<MetronomeBeats> createState() => _MetronomeBeatsState();
}

class _MetronomeBeatsState extends ConsumerState<MetronomeBeats>
    with SingleTickerProviderStateMixin {
  late AnimationController _animationController;
  late Animation<double> _animation;

  @override
  void initState() {
    super.initState();
    _animationController = AnimationController(
      duration: const Duration(milliseconds: 100),
      vsync: this,
    );
    _animation = Tween<double>(begin: 0.0, end: 1.0).animate(
      CurvedAnimation(
        parent: _animationController,
        curve: Curves.easeOut,
      ),
    );
  }

  @override
  void dispose() {
    _animationController.dispose();
    super.dispose();
  }

  double bloomWidthandHeight(int idx) {
    // bloom 효과의 크기 결정
    bool currItemBase4 = ref.watch(currItemBase4Provider); // 4분음표 기준인지

    if (!currItemBase4) {
      // 4분음표 기준이 아닐 때
      if (idx % 3 == 0) {
        return 90; // 점4분음표 기준일 때
      } else {
        return 60; // 나머지 박자일 때
      } // 점4분음표 기준일 때
    }
    return 90; // 기본값
  }

  List<dynamic> metronomeItemStyle(int idx) {
    // 메트로놈 아이템 스타일 정의
    String currBeatPattern = ref.watch(currBeatPatternProvider); // 현재 박자 패턴
    bool isSelected = (idx == ref.watch(currItemProvider).toInt()); // 현재 박자인지
    bool currItemBase4 = ref.watch(currItemBase4Provider); // 4분음표 기준인지

    double width = 80;
    double height = 80;
    Color color = isSelected
        ? const Color(0xffF28C79)
        : const Color.fromARGB(255, 236, 122, 102);
    Color borderColor = isSelected
        ? Colors.transparent
        : const Color.fromARGB(255, 125, 71, 61);

    if (!currItemBase4) {
      if (currBeatPattern == 'dot_quarter') {
        if (idx % 3 != 0) {
          width = 50;
          height = 50;
          color = isSelected
              ? const Color(0xffF28C79)
              : const Color.fromARGB(255, 108, 107, 106);
          borderColor = isSelected ? Colors.transparent : Colors.black45;
        }
      } else if (currBeatPattern == "three") {
        if (idx % 3 != 0) {
          width = 50;
          height = 50;
        }
      } else if (currBeatPattern == 'three_2') {
        if (idx % 3 != 0) {
          width = 50;
          height = 50;
          color = isSelected
              ? const Color(0xffF28C79)
              : (idx % 3 == 2)
                  ? const Color.fromARGB(255, 236, 122, 102)
                  : const Color.fromARGB(255, 108, 107, 106);
          borderColor = isSelected
              ? Colors.transparent
              : (idx % 3 == 2)
                  ? const Color.fromARGB(255, 125, 71, 61)
                  : Colors.black45;
        }
      }
    }
    return [
      width,
      height,
      color,
      borderColor,
    ];
  }

  Widget metronomeItem(int idx) {
    // 메트로놈 아이템 생성 - 원 하나
    final currItem = ref.watch(currItemProvider).toInt();
    final isPlaying = ref.watch(isPlayingProvider);
    final isSelected = (idx == currItem);
    final shouldShowBloom = isSelected && isPlaying;
    List<dynamic> style = metronomeItemStyle(idx);

    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 50),
      child: Stack(
        clipBehavior: Clip.none,
        children: [
          if (shouldShowBloom) // bloom 효과 대상 인지 - bloom효과 방식 상황에 따라 정의
            Positioned(
              // 4분음표 기준일 때도 고정된 bloom 효과
              left: -5,
              top: -5,
              child: Container(
                width: bloomWidthandHeight(idx),
                height: bloomWidthandHeight(idx),
                decoration: BoxDecoration(
                  color: const Color(0xffB95D4C),
                  borderRadius: BorderRadius.circular(100),
                  boxShadow: [
                    BoxShadow(
                      color: const Color(0xffD97D6C).withOpacity(0.7),
                      blurRadius: 5,
                      spreadRadius: 5,
                    ),
                  ],
                ),
              ),
            ),
          Container(
            // 메트로놈 원 기본 스타일 정의
            width: style[0],
            height: style[1],
            decoration: BoxDecoration(
              color: style[2],
              borderRadius: BorderRadius.circular(100),
              border: Border.all(
                color: style[3],
                width: 4,
              ),
            ),
          ),
        ],
      ),
    );
  }

  List<Widget> generateMetronomeItems(int count) {
    // 메트로놈 아이템 생성
    List<Widget> items = List.generate(
        count, (index) => metronomeItem(index)); // List.generate 이용해서 자동 생성

    String timeSignature = ref.watch(selectedTimeSignatureProvider); // 현재 박자

    if (timeSignature != '9/8' && timeSignature != '12/8') {
      // 일렬로 출력
      return [
        Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: items,
        )
      ];
    } else {
      if (timeSignature == '9/8') {
        // 2열로 출력 3 + 6
        return [
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: items.sublist(0, 3),
          ),
          const SizedBox(height: 20),
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: items.sublist(3, 9),
          ),
        ];
      } else {
        // 12/8 박자일 때 - 2열로 출력 6 + 6
        return [
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: items.sublist(0, 6),
          ),
          const SizedBox(height: 20),
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: items.sublist(6, 12),
          ),
        ];
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    // 최종 build 메서드
    String timeSignature = ref.watch(selectedTimeSignatureProvider); // 현재 박자
    int count =
        int.parse(timeSignature.split('/')[0]); // 현재 박자에 따라 메트로놈 원 개수 결정

    return Container(
      decoration: const BoxDecoration(
        color: Color(0xff424242),
      ),
      child: Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: generateMetronomeItems(count),
        ),
      ),
    );
  }
}
