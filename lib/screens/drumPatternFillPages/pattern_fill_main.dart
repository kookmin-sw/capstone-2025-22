import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:capstone_2025/screens/drumPatternFillPages/widgets/innerShadow.dart';
import 'package:capstone_2025/screens/drumPatternFillPages/widgets/linedText.dart';
import 'package:capstone_2025/screens/drumPatternFillPages/pattern_fill_screen.dart';

class PatternFillMain extends StatefulWidget {
  const PatternFillMain({super.key});

  @override
  State<PatternFillMain> createState() => _PatternFillMainState();
}

class _PatternFillMainState extends State<PatternFillMain> {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Expanded(
        child: Padding(
          padding: const EdgeInsets.all(20),
          child: Column(
            // crossAxisAlignment: CrossAxisAlignment.center,
            children: [
              Expanded(
                flex: 3,
                child: Center(
                  child: linedText("LEVEL", 35, Colors.black54, Colors.white,
                      5), // 테두리 있는 텍스트
                ),
              ),
              Spacer(flex: 1),
              Expanded(
                flex: 20,
                child: SingleChildScrollView(
                  child: Column(
                    children: [
                      clickedListItem(context, 1, true, false),
                      clickedListItem(context, 2, false, false),
                      clickedListItem(context, 3, false, true),
                      clickedListItem(context, 4, false, true),
                      clickedListItem(context, 5, false, true),
                      clickedListItem(context, 6, false, true),
                      clickedListItem(context, 7, false, true),
                      clickedListItem(context, 8, false, true),
                      clickedListItem(context, 9, false, true),
                    ],
                  ),
                ),
              )
            ],
          ),
        ),
      ),
    );
  }

  Widget clickedListItem(BuildContext context, int index, bool isLevelCleared,
      bool isLevelLocked) {
    // List item 클릭 시 동작
    return InkWell(
      onTap: () {
        if (!isLevelLocked) {
          // 레벨 잠금 해제
          Navigator.push(
            context,
            MaterialPageRoute(builder: (context) => PatternFillScreen()),
          );
        }
      },
      child: patternFillList(
        context,
        index,
        isLevelCleared,
        isLevelLocked,
      ),
    );
  }

  Widget patternFillList(BuildContext context, int index, bool isLevelCleared,
      bool isLevelLocked) {
    double containerHeight = 50;
    double containerWidth = MediaQuery.of(context).size.width * 0.67;
    double borderRadius = 13;
    double fontSize = 19;
    int score = 95;

    return Padding(
      padding: const EdgeInsets.only(bottom: 10),
      child: Stack(children: [
        InnerShadow(
          // List item 내부 그림자
          shadowColor: isLevelCleared
              ? Color.fromARGB(255, 252, 242, 209)
              : isLevelLocked
                  ? const Color.fromARGB(255, 177, 177, 177)
                  : Color.fromARGB(255, 238, 159, 145).withOpacity(0.5),
          blur: 7, // 그림자 흐림 정도
          offset: Offset(0, 0),
          borderRadius: BorderRadius.circular(borderRadius),
          child: Container(
            width: containerWidth,
            height: containerHeight,
            decoration: BoxDecoration(
              // List item 배경색
              color: isLevelCleared
                  ? Color(0xFFFFE89B)
                  : isLevelLocked
                      ? Color(0xFF949494)
                      : Color(0xFFC76A59),
              borderRadius: BorderRadius.circular(borderRadius),
            ),
            child: Padding(
              padding: const EdgeInsets.all(5.0),
              child: Row(
                // List item 내부 요소 - 텍스트 점수 등
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Padding(
                    padding: const EdgeInsets.only(left: 35),
                    child: linedText(
                      "$index. Basic Pattern $index",
                      fontSize,
                      isLevelCleared
                          ? Color(0xFFF1B45F)
                          : isLevelLocked
                              ? Color(0xff797979)
                              : Color(0xffB95D4C),
                      Colors.white,
                      4,
                    ),
                  ),
                  Padding(
                    padding: const EdgeInsets.only(right: 35),
                    child: Container(
                      width: 50,
                      child: isLevelCleared
                          ? linedText(
                              '$score점',
                              fontSize,
                              Color(0xFFF1B45F),
                              Colors.white,
                              4,
                            )
                          : isLevelLocked
                              ? Icon(
                                  Icons.lock,
                                  size: 30,
                                  color: Colors.white,
                                )
                              : linedText(
                                  '-',
                                  fontSize,
                                  Color(0xffB95D4C),
                                  Colors.white,
                                  4,
                                ),
                    ),
                  ),
                ],
              ),
            ),
          ),
        ),
        Container(
          // List item 테두리
          width: containerWidth,
          height: containerHeight,
          decoration: BoxDecoration(
            border: Border.all(
              color: isLevelCleared
                  ? Color(0xFFF1B45F)
                  : isLevelLocked
                      ? Color(0xff797979)
                      : Color(0xffB95D4C),
              width: 3, // 테두리 두께
            ),
            borderRadius: BorderRadius.circular(borderRadius),
          ),
        ),
      ]),
    );
  }
}
