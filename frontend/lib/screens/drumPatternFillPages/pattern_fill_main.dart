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
      body: Column(
        children: [
          Expanded(
            child: Padding(
              padding: const EdgeInsets.all(20),
              child: Column(
                // crossAxisAlignment: CrossAxisAlignment.center,
                children: [
                  Expanded(
                    flex: 3,
                    child: Center(
                      child: linedText("LEVEL", 35, Colors.black45,
                          Colors.white, 5.5), // 테두리 있는 텍스트
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
        ],
      ),
    );
  }

  Widget clickedListItem(BuildContext context, int index, bool isLevelCleared,
      bool isLevelLocked) {
    // List item 클릭 시 동작
    return InkWell(
      onTap: () {
        if (!isLevelLocked) {
          openModal(context, index);
          // 레벨 클리어 시 모달 열기
          // 레벨 잠금 해제

          // );
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

  Widget modalBtn(BuildContext context, String text, Color backgroundColor,
      bool isTextblack) {
    // 모달 버튼
    return Container(
      padding: const EdgeInsets.all(5),
      margin: const EdgeInsets.only(left: 0, right: 0),
      width: 155,
      // MediaQuery.of(context).size.width * 0.168,
      height: 57,
      // MediaQuery.of(context).size.height * 0.135,
      alignment: Alignment.center,
      decoration: BoxDecoration(
        color: backgroundColor,
        borderRadius: BorderRadius.circular(15),
      ),
      child: Text(text,
          style: TextStyle(
              color: isTextblack ? Colors.black : Colors.white,
              fontSize: 15,
              fontWeight: FontWeight.w500)),
    );
  }

  void openModal(BuildContext context, int index) {
    // 모달 열기
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        alignment: Alignment.center,
        insetPadding: EdgeInsets.symmetric(horizontal: 0, vertical: 0),
        contentPadding: EdgeInsets.only(top: 20, bottom: 10),
        content: SizedBox(
          width: 380,
          height: 170,
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Padding(
                padding: const EdgeInsets.only(top: 15, bottom: 5),
                child: Text(
                  'Basic Pattern $index',
                  textAlign: TextAlign.center,
                  style: TextStyle(
                    fontSize: 21,
                    fontWeight: FontWeight.bold,
                    color: Color(0xFF4A4A4A),
                  ),
                ),
              ),
              Text(
                '연습을 시작하시겠습니까?',
                textAlign: TextAlign.center,
                style: TextStyle(
                  fontSize: 18,
                  color: Color(0xFF4A4A4A),
                ),
              ),
              const SizedBox(height: 20),
              Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  TextButton(
                    onPressed: () {
                      Navigator.of(context).pop();
                    },
                    child: modalBtn(context, '취소',
                        Color.fromARGB(255, 205, 203, 202), true),
                  ),
                  TextButton(
                    onPressed: () {
                      Navigator.push(
                          context,
                          MaterialPageRoute(
                              builder: (context) => PatternFillScreen(
                                  title: 'Basic Pattern $index')));
                    },
                    child: modalBtn(context, '확인', Color(0xffD97D6C), false),
                  ),
                ],
              ),
            ],
          ),
        ),
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
                    child: SizedBox(
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
