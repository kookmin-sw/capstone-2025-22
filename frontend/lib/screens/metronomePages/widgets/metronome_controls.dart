import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:capstone_2025/providers/metronome_provider.dart';
import 'package:flutter/material.dart';

// 재사용 가능한 모달 함수
void showCustomModal({
  required BuildContext context,
  required Widget content,
  Offset? anchorPoint,
}) {
  showDialog(
    context: context,
    barrierColor: Colors.black54,
    anchorPoint: anchorPoint ?? const Offset(100, 300),
    builder: (BuildContext context) {
      return Dialog(
        insetPadding: EdgeInsets.zero,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(15),
        ),
        child: IntrinsicWidth(
          child: Container(
            constraints: const BoxConstraints(
              maxHeight: 400,
            ),
            padding: const EdgeInsets.symmetric(vertical: 15, horizontal: 20),
            decoration: BoxDecoration(
              color: Colors.white,
              borderRadius: BorderRadius.circular(15),
            ),
            child: SingleChildScrollView(
              child: content,
            ),
          ),
        ),
      );
    },
  );
}

class MetronomeControls extends ConsumerStatefulWidget {
  final GlobalKey metronomeHeaderKey;
  const MetronomeControls({super.key, required this.metronomeHeaderKey});

  @override
  ConsumerState<MetronomeControls> createState() => _MetronomeControlsState();
}

class _MetronomeControlsState extends ConsumerState<MetronomeControls> {
  late final TextEditingController bpmController;
  late final FocusNode bpmFocusNode;
  late Image currBeatPatternImg;
  double beatPatternImgSize = 60; // 비트 패턴 이미지 크기

  @override
  void initState() {
    // 위젯이 생성될 때 호출
    super.initState();

    bpmFocusNode = FocusNode();
    currBeatPatternImg = Image.asset('assets/images/notes/quarter.png');
  }

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();

    bpmController = TextEditingController(
      text: ref.watch(currBPMProvider).toString(),
    );

    bpmFocusNode.addListener(() {
      setState(() {
        if (bpmFocusNode.hasFocus) {
          bpmController.clear();
        } else {
          final bpm = int.tryParse(bpmController.text);
          if (bpm != null && bpm >= 10 && bpm <= 400) {
            ref.read(currBPMProvider.notifier).state = bpm;
            if (ref.read(isPlayingProvider)) {
              (widget.metronomeHeaderKey.currentState as dynamic)
                  ?.updateMetronomeTiming();
            }
          }
          bpmController.text = ref.watch(currBPMProvider).toString();
        }
      });
    });
  }

  @override
  void dispose() {
    bpmController.dispose();
    bpmFocusNode.dispose();
    super.dispose();
  }

  void changeTimeSignature(String timeSignature) {
    // 박자 선택 시 수행될 함수
    // 선택된 박자값 갱신
    ref.read(selectedTimeSignatureProvider.notifier).state = timeSignature;

    // currItemBase4Provider 갱신
    if (timeSignature == '3/8' ||
        timeSignature == '6/8' ||
        timeSignature == '9/8' ||
        timeSignature == '12/8') {
      ref.read(currItemBase4Provider.notifier).state = false;
    } else {
      ref.read(currItemBase4Provider.notifier).state = true;
    }

    // 비트 패턴 이미지 변경 - default로
    if (ref.watch(currItemBase4Provider)) {
      // 4분음표 기준일 때
      ref.read(currBeatPatternProvider.notifier).state = 'quarter';
      changeBeatPatternImg('quarter');
    } else {
      // 8분음표 기준일 때
      ref.read(currBeatPatternProvider.notifier).state = 'dot_quarter';
      changeBeatPatternImg('dot_quarter');
    }
    (widget.metronomeHeaderKey.currentState as dynamic)?.stopMetronome();
  }

  void changeBeatPattern(String beatPattern) {
    // 비트 패턴 변경 시 수행될 함수
    // 선택된 비트 패턴값 갱신
    ref.read(currBeatPatternProvider.notifier).state = beatPattern;
    setState(() {
      changeBeatPatternImg(beatPattern); // 비트 패턴 이미지 변경
    });
    (widget.metronomeHeaderKey.currentState as dynamic)
        ?.updateMetronomeTiming();
  }

  void changeBeatPatternImg(String beatPattern) {
    // 비트 패턴 이미지 변경
    switch (beatPattern) {
      case 'quarter':
        currBeatPatternImg = Image.asset('assets/images/notes/quarter.png');
        beatPatternImgSize = 50;
        break;
      case 'two':
        currBeatPatternImg = Image.asset('assets/images/notes/two.png');
        beatPatternImgSize = 45;
        break;
      case 'triplet':
        currBeatPatternImg = Image.asset('assets/images/notes/triplet.png');
        beatPatternImgSize = 63;
        break;
      case 'triplet2':
        currBeatPatternImg = Image.asset('assets/images/notes/triplet2.png');
        beatPatternImgSize = 63;
        break;
      case 'four':
        currBeatPatternImg = Image.asset('assets/images/notes/four.png');
        beatPatternImgSize = 40;
        break;
      case 'four_2':
        currBeatPatternImg = Image.asset('assets/images/notes/four_2.png');
        beatPatternImgSize = 40;
        break;
      case 'dot_quarter':
        currBeatPatternImg = Image.asset('assets/images/notes/dot_quarter.png');
        beatPatternImgSize = 50;
        break;
      case 'three':
        currBeatPatternImg = Image.asset('assets/images/notes/three.png');
        beatPatternImgSize = 45;
        break;
      case 'three_2':
        currBeatPatternImg = Image.asset('assets/images/notes/three_2.png');
        beatPatternImgSize = 45;
        break;
    }
  }

  Widget _buildModalItem(
      // 모달 내 항목 위젯 생성
      BuildContext context,
      dynamic data,
      VoidCallback action) {
    return InkWell(
      onTap: () => {Navigator.pop(context), action()},
      child: Padding(
        padding: const EdgeInsets.symmetric(vertical: 12),
        child: data is String
            ? Text(
                data,
                style:
                    const TextStyle(fontSize: 18, fontWeight: FontWeight.w500),
              )
            : data is Image
                ? data
                : const SizedBox(),
      ),
    );
  }

  void _showBeatSelectModal(BuildContext context) {
    // 박자 선택 모달
    showCustomModal(
      context: context,
      content: Row(
        mainAxisAlignment: MainAxisAlignment.spaceAround,
        children: [
          Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              _buildModalItem(
                context,
                "1/4",
                () => changeTimeSignature("1/4"),
              ),
              _buildModalItem(
                context,
                "2/4",
                () => changeTimeSignature("2/4"),
              ),
              _buildModalItem(
                context,
                "3/4",
                () => changeTimeSignature("3/4"),
              ),
              _buildModalItem(
                context,
                "4/4",
                () => changeTimeSignature("4/4"),
              ),
            ],
          ),
          SizedBox(
            height: 185,
            child: VerticalDivider(
              thickness: 4,
              color: const Color(0xff494949).withOpacity(0.5),
            ),
          ),
          Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              _buildModalItem(
                context,
                "3/8",
                () => changeTimeSignature("3/8"),
              ),
              _buildModalItem(
                context,
                "6/8",
                () => changeTimeSignature("6/8"),
              ),
              _buildModalItem(
                context,
                "9/8",
                () => changeTimeSignature("9/8"),
              ),
              _buildModalItem(
                context,
                "12/8",
                () => changeTimeSignature("12/8"),
              ),
            ],
          ),
        ],
      ),
    );
  }

  void _showBeatPatternModal(BuildContext context) {
    // 비트 패턴 변경 모달
    bool currItemBase4 = ref.watch(currItemBase4Provider);

    if (!currItemBase4) {
      // 8분 음표 기준 박자일 때
      showCustomModal(
        context: context,
        content: Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const SizedBox(width: 10),
            _buildModalItem(
              context,
              Image.asset(
                'assets/images/notes/dot_quarter.png',
                width: 25,
                fit: BoxFit.contain,
              ),
              () => changeBeatPattern("dot_quarter"),
            ),
            const SizedBox(width: 60),
            _buildModalItem(
              context,
              Image.asset(
                'assets/images/notes/three.png',
                width: 65,
                fit: BoxFit.contain,
              ),
              () => changeBeatPattern("three"),
            ),
            const SizedBox(width: 65),
            _buildModalItem(
              context,
              Image.asset(
                'assets/images/notes/three_2.png',
                width: 65,
                fit: BoxFit.contain,
              ),
              () => changeBeatPattern("three_2"),
            ),
            const SizedBox(width: 10),
          ],
        ),
      );
    } else {
      showCustomModal(
        context: context,
        content: Column(
          mainAxisAlignment: MainAxisAlignment.spaceEvenly,
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              crossAxisAlignment: CrossAxisAlignment.end,
              children: [
                _buildModalItem(
                  context,
                  Image.asset(
                    'assets/images/notes/quarter.png',
                    width: 17,
                    fit: BoxFit.contain,
                  ),
                  () => changeBeatPattern("quarter"),
                ),
                const SizedBox(width: 90),
                _buildModalItem(
                  context,
                  Image.asset(
                    'assets/images/notes/two.png',
                    width: 40,
                    fit: BoxFit.contain,
                  ),
                  () => changeBeatPattern("two"),
                ),
                const SizedBox(width: 75),
                _buildModalItem(
                  context,
                  Image.asset(
                    'assets/images/notes/triplet.png',
                    width: 60,
                    fit: BoxFit.contain,
                  ),
                  () => changeBeatPattern("triplet"),
                ),
              ],
            ),
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              crossAxisAlignment: CrossAxisAlignment.end,
              children: [
                const SizedBox(width: 10),
                _buildModalItem(
                  context,
                  Image.asset(
                    'assets/images/notes/triplet2.png',
                    width: 50,
                    fit: BoxFit.contain,
                  ),
                  () => changeBeatPattern("triplet2"),
                ),
                const SizedBox(width: 50),
                _buildModalItem(
                  context,
                  Image.asset(
                    'assets/images/notes/four.png',
                    width: 80,
                    fit: BoxFit.contain,
                  ),
                  () => changeBeatPattern("four"),
                ),
                const SizedBox(width: 50),
                _buildModalItem(
                  context,
                  Image.asset(
                    'assets/images/notes/four_2.png',
                    width: 80,
                    fit: BoxFit.contain,
                  ),
                  () => changeBeatPattern("four_2"),
                ),
                const SizedBox(width: 10),
              ],
            ),
          ],
        ),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    final currBPM = ref.watch(currBPMProvider);
    return Row(
      children: [
        const Spacer(flex: 2),
        Expanded(
          // 박자 선택 버튼
          flex: 2,
          child: TextButton(
            onPressed: () => _showBeatSelectModal(context),
            child: Text(
              ref.watch(selectedTimeSignatureProvider).toString(),
              style: const TextStyle(
                fontSize: 35,
                fontWeight: FontWeight.w600,
                color: Color(0xff424242),
              ),
            ),
          ),
        ),
        const Spacer(flex: 1),
        Expanded(
          // BPM 조절 필드
          flex: 4,
          child: Padding(
            padding: const EdgeInsets.symmetric(vertical: 17),
            child: Container(
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(20),
                boxShadow: [
                  BoxShadow(
                    color: Colors.grey.withOpacity(0.5),
                    blurRadius: 10,
                    spreadRadius: 1,
                    offset: const Offset(3, 3),
                  ),
                ],
              ),
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  const Text(
                    'BPM',
                    style: TextStyle(
                      fontSize: 25,
                      fontWeight: FontWeight.w400,
                    ),
                  ),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                    children: [
                      IconButton(
                        // BPM 감소 버튼
                        onPressed: currBPM <= 10
                            ? null
                            : () {
                                ref.read(currBPMProvider.notifier).state--;
                                bpmController.text =
                                    ref.watch(currBPMProvider).toString();
                                if (ref.read(isPlayingProvider)) {
                                  (widget.metronomeHeaderKey.currentState
                                          as dynamic)
                                      ?.updateMetronomeTiming();
                                }
                              },
                        icon: const Icon(Icons.remove, size: 50),
                      ),
                      SizedBox(
                        width: 100,
                        height: 50,
                        child: TextField(
                          // BPM 입력 필드
                          controller: bpmController,
                          focusNode: bpmFocusNode,
                          textAlign: TextAlign.center,
                          showCursor: false,
                          maxLength: 3,
                          buildCounter: (context,
                                  {required currentLength,
                                  required isFocused,
                                  maxLength}) =>
                              null,
                          decoration: const InputDecoration(
                            border: InputBorder.none,
                            counterText: '',
                          ),
                          style: TextStyle(
                            fontSize: 50,
                            fontWeight: FontWeight.bold,
                            color: bpmFocusNode.hasFocus // 포커스 여부에 따라 색상 변경
                                ? const Color(0xffD97D6C)
                                : const Color(0xff424242),
                          ),
                          keyboardType: TextInputType.number,
                          onSubmitted: (value) {
                            // BPM 입력 후 엔터키 눌렀을 때
                            final bpm = int.tryParse(value);
                            if (bpm != null && bpm >= 10 && bpm <= 400) {
                              // 유효한 BPM 값일 때
                              ref.read(currBPMProvider.notifier).state = bpm;
                              if (ref.read(isPlayingProvider)) {
                                (widget.metronomeHeaderKey.currentState
                                        as dynamic)
                                    ?.updateMetronomeTiming();
                              }
                            } else {
                              // 유효하지 않은 값일 때
                              bpmController.text =
                                  ref.watch(currBPMProvider).toString();
                            }
                          },
                        ),
                      ),
                      IconButton(
                        // BPM 증가 버튼
                        onPressed: currBPM >= 400
                            ? null
                            : () {
                                ref.read(currBPMProvider.notifier).state++;
                                bpmController.text =
                                    ref.watch(currBPMProvider).toString();
                                if (ref.read(isPlayingProvider)) {
                                  (widget.metronomeHeaderKey.currentState
                                          as dynamic)
                                      ?.updateMetronomeTiming();
                                }
                              },
                        icon: const Icon(Icons.add, size: 50),
                      ),
                    ],
                  ),
                ],
              ),
            ),
          ),
        ),
        const Spacer(flex: 1),
        Expanded(
          // 비트 패턴 선택 버튼
          flex: 2,
          child: InkWell(
            onTap: () => _showBeatPatternModal(context),
            child: Container(
              child: currBeatPatternImg,
              height: beatPatternImgSize,
            ),
          ),
        ),
        const Spacer(flex: 2),
      ],
    );
  }
}
