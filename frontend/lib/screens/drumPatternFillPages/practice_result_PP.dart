import 'package:capstone_2025/screens/drumPatternFillPages/pattern_fill_screen.dart';
import 'package:capstone_2025/widgets/innerShadow.dart';
import 'package:capstone_2025/widgets/linedText.dart';
import 'package:capstone_2025/screens/mainPages/navigation_screens.dart';
import 'package:capstone_2025/widgets/openSheetModal.dart';
import 'package:flutter/material.dart';
import 'package:flutter_screenutil/flutter_screenutil.dart';

class PracticeResultPP extends StatefulWidget {
  const PracticeResultPP(
      {super.key,
      required this.idx,
      required this.score,
      required this.xmlDataString,
      required this.practiceInfo});

  final int idx; // 패턴 번호
  final int score; // 점수
  final String xmlDataString; // musicxml
  final List<Map<String, dynamic>> practiceInfo; // 1차 채점 결과

  @override
  State<PracticeResultPP> createState() => _PracticeResultPPState();
}

class _PracticeResultPPState extends State<PracticeResultPP> {
  late int idx = widget.idx; // 패턴 번호
  late int score = widget.score; // 점수
  late bool lvCleared = (widget.score >= 80); // 레벨 클리어 여부

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: LayoutBuilder(
        builder: (context, constraints) {
          return SingleChildScrollView(
            child: ConstrainedBox(
              constraints: BoxConstraints(minHeight: constraints.maxHeight),
              child: IntrinsicHeight(
                child: Container(
                  color: Color(0xFFF2F1F3),
                  child: Padding(
                    padding: EdgeInsets.only(left: 10.w, top: 30.h),
                    child: Stack(
                      children: [
                        IconButton(
                          // 홈 버튼
                          onPressed: () => {
                            Navigator.pushReplacement(
                                context,
                                MaterialPageRoute(
                                  builder: (context) =>
                                      NavigationScreens(firstSelectedIndex: 4),
                                ))
                          },
                          icon: Icon(
                            Icons.home_filled,
                            size: 14.sp,
                          ),
                        ),
                        Padding(
                          // 결과창 padding
                          padding: EdgeInsets.symmetric(
                            // padding으로 사이즈 조절
                            vertical: MediaQuery.of(context).size.height * 0.08,
                            horizontal:
                                MediaQuery.of(context).size.width * 0.12,
                          ),
                          child: Container(
                            // 결과창 container
                            decoration: BoxDecoration(
                              borderRadius: BorderRadius.circular(30),
                              boxShadow: [
                                BoxShadow(
                                  color: Colors.black.withOpacity(0.45),
                                  blurRadius: 10,
                                  spreadRadius: 2,
                                  offset: Offset(0, 0),
                                ),
                              ],
                            ),
                            child: Stack(
                              children: [
                                // 내부 그림자, 내부 요소 구분
                                InnerShadow(
                                  shadowColor:
                                      const Color.fromARGB(255, 244, 244, 244)
                                          .withOpacity(0.7),
                                  blur: 6,
                                  offset: Offset(0, 0),
                                  borderRadius: BorderRadius.circular(30),
                                  child: Container(
                                    decoration: BoxDecoration(
                                      color: Color(0xFFD9D9D9),
                                      borderRadius: BorderRadius.circular(30),
                                    ),
                                  ),
                                ),
                                Padding(
                                  padding: EdgeInsets.symmetric(
                                      vertical: 20.h, horizontal: 20.w),
                                  child: Column(
                                    mainAxisAlignment: MainAxisAlignment.center,
                                    // 결과창 내부 요소들
                                    children: [
                                      Row(
                                        mainAxisAlignment:
                                            MainAxisAlignment.spaceBetween,
                                        // 성공 여부 텍스트 및 점수 row
                                        children: [
                                          Padding(
                                            padding: EdgeInsets.only(
                                                left: lvCleared ? 0 : 10.w),
                                            child: Column(
                                              // 성공 여부 텍스트
                                              children: [
                                                SizedBox(
                                                  child: linedText(
                                                      "Basic Pattern $idx",
                                                      12.sp,
                                                      Colors.black
                                                          .withOpacity(0.3),
                                                      Colors.white,
                                                      7),
                                                ),
                                                SizedBox(height: 10.h),
                                                lvCleared
                                                    ? Stack(children: [
                                                        Text(
                                                          "CLEAR",
                                                          style: TextStyle(
                                                            fontSize: 32.sp,
                                                            fontWeight:
                                                                FontWeight.bold,
                                                            color: Colors
                                                                .transparent,
                                                            shadows: [
                                                              Shadow(
                                                                offset: Offset(
                                                                    4, 4),
                                                                blurRadius: 35,
                                                                color: Colors
                                                                    .black
                                                                    .withOpacity(
                                                                        0.7),
                                                              ),
                                                            ],
                                                            height: 1,
                                                          ),
                                                        ),
                                                        linedText(
                                                          "CLEAR",
                                                          32.sp,
                                                          Color(0xffB95D4C),
                                                          Color(0xffFD9B8A),
                                                          9.5,
                                                        ),
                                                      ])
                                                    : Stack(children: [
                                                        Text(
                                                          "FAIL",
                                                          style: TextStyle(
                                                            fontSize: 35.sp,
                                                            fontWeight:
                                                                FontWeight.bold,
                                                            color: Colors
                                                                .transparent,
                                                            shadows: [
                                                              Shadow(
                                                                offset: Offset(
                                                                    4, 4),
                                                                blurRadius: 35,
                                                                color: Colors
                                                                    .black
                                                                    .withOpacity(
                                                                        0.7),
                                                              ),
                                                            ],
                                                            height: 1,
                                                          ),
                                                        ),
                                                        linedText(
                                                            "FAIL",
                                                            35.sp,
                                                            Color(0xff4C7FB9),
                                                            Color(0xff8ABCFD),
                                                            9.5),
                                                      ]),
                                              ],
                                            ),
                                          ),
                                          Container(
                                            decoration: BoxDecoration(
                                              color: Colors.white,
                                              border: Border.all(
                                                color: Colors.black26,
                                                width: 5,
                                              ),
                                              borderRadius:
                                                  BorderRadius.circular(38),
                                            ),
                                            child: Padding(
                                              padding: EdgeInsets.symmetric(
                                                  vertical: 10.h,
                                                  horizontal: 28.w),
                                              child: Column(children: [
                                                linedText(
                                                  'SCORE',
                                                  11.sp,
                                                  Colors.black26,
                                                  Colors.white,
                                                  4.5,
                                                ),
                                                SizedBox(height: 10.h),
                                                Stack(children: [
                                                  Text(
                                                    "$score",
                                                    style: TextStyle(
                                                      height: 1,
                                                      fontSize: 35.sp,
                                                      color: Colors.transparent,
                                                      shadows: [
                                                        Shadow(
                                                          offset: Offset(4, 4),
                                                          blurRadius: 35,
                                                          color: Colors.black
                                                              .withOpacity(0.7),
                                                        ),
                                                      ],
                                                    ),
                                                  ),
                                                  lvCleared
                                                      ? linedText(
                                                          '$score',
                                                          35.sp,
                                                          Color(0xffF1B45F),
                                                          Color(0xffFFE89B),
                                                          9.5,
                                                        )
                                                      : linedText(
                                                          '$score',
                                                          35.sp,
                                                          Color(0xff949494),
                                                          Color(0xffD9D9D9),
                                                          9.5,
                                                        ),
                                                ])
                                              ]),
                                            ),
                                          ),
                                        ],
                                      ),
                                      SizedBox(height: 10.h),
                                      ButtonForm(
                                        btnName: "상세 기록 확인하기",
                                        buttonColor: Color(0xff949494),
                                        borderColor:
                                            Color.fromARGB(255, 104, 104, 104),
                                        shadowColor:
                                            Color.fromARGB(255, 177, 177, 177),
                                        width:
                                            MediaQuery.of(context).size.width *
                                                0.65,
                                        clickedFunc: () {
                                          openMusicSheet(
                                            context: context,
                                            xmlDataString: widget.xmlDataString,
                                            practiceInfo: widget.practiceInfo,
                                            isPatternMode: true,
                                          );
                                        },
                                        btnIcon:
                                            Icons.insert_drive_file_rounded,
                                      ),
                                      SizedBox(height: 20.h),
                                      Row(
                                        children: [
                                          Expanded(
                                            flex: 12,
                                            child: Material(
                                              type: MaterialType.transparency,
                                              child: ButtonForm(
                                                btnName: '다시 하기',
                                                buttonColor: Color(0xffD97D6C),
                                                borderColor: Color(0xffC76A59),
                                                shadowColor: Color.fromARGB(
                                                        255, 248, 180, 168)
                                                    .withOpacity(0.5),
                                                clickedFunc: () {
                                                  openModal(context, idx);
                                                },
                                              ),
                                            ),
                                          ),
                                          SizedBox(width: 5.w),
                                          Expanded(
                                            flex: 10,
                                            child: Material(
                                              type: MaterialType.transparency,
                                              child: ButtonForm(
                                                btnName: '목록',
                                                buttonColor: Color(0xff8ABCFD),
                                                borderColor: Color(0xff4C7FB9),
                                                shadowColor: Color.fromARGB(
                                                    255, 196, 213, 237),
                                                btnIcon: Icons.list,
                                                clickedFunc: () {
                                                  Navigator.pushReplacement(
                                                    context,
                                                    MaterialPageRoute(
                                                      builder: (context) =>
                                                          NavigationScreens(
                                                        firstSelectedIndex: 2,
                                                      ),
                                                    ),
                                                  );
                                                },
                                              ),
                                            ),
                                          ),
                                          SizedBox(width: 5.w),
                                          Expanded(
                                            flex: 12,
                                            child: Stack(children: [
                                              ButtonForm(
                                                btnName: '다음 단계',
                                                buttonColor: Color(0xFFFFE89B),
                                                borderColor: Color(0xFFF1B45F),
                                                shadowColor: Color.fromARGB(
                                                    255, 243, 235, 211),
                                                clickedFunc: lvCleared
                                                    ? () {
                                                        Navigator
                                                            .pushReplacement(
                                                          context,
                                                          MaterialPageRoute(
                                                            builder: (context) =>
                                                                PatternFillScreen(
                                                              index: idx + 1,
                                                            ),
                                                          ),
                                                        );
                                                      }
                                                    : null,
                                              ),
                                              if (!lvCleared)
                                                Container(
                                                  width: double.infinity,
                                                  height: 70.h,
                                                  decoration: BoxDecoration(
                                                    color: Colors.black26,
                                                    borderRadius:
                                                        BorderRadius.circular(
                                                            20),
                                                  ),
                                                  child: Center(
                                                      child: Icon(
                                                    Icons.lock,
                                                    size: 10.spMax,
                                                    color: Colors.white,
                                                  )),
                                                ),
                                            ]),
                                          ),
                                        ],
                                      )
                                    ],
                                  ),
                                ),
                              ],
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
            ),
          );
        },
      ),
    );
  }
}

class ButtonForm extends StatelessWidget {
  ButtonForm({
    super.key,
    required this.btnName,
    required this.buttonColor,
    required this.borderColor,
    required this.clickedFunc,
    required this.shadowColor,
    this.width = double.infinity,
    this.btnIcon = null,
  });

  final String btnName;
  final Color buttonColor;
  final Color borderColor;
  final Color shadowColor;
  final IconData? btnIcon;
  final VoidCallback? clickedFunc;
  final double width;

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      height: 70.h,
      width: width,
      child: Stack(
        children: [
          InnerShadow(
            shadowColor: shadowColor,
            blur: 7,
            offset: Offset(0, 0),
            borderRadius: BorderRadius.circular(20),
            child: Container(
              decoration: BoxDecoration(
                color: buttonColor,
                borderRadius: BorderRadius.circular(20),
              ),
              padding: EdgeInsets.symmetric(horizontal: 6.w),
              child: Center(
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    if (btnIcon != null)
                      Icon(
                        btnIcon,
                        size: 10.sp,
                        color: Colors.white,
                      ),
                    if (btnIcon != null) SizedBox(width: 15),
                    linedText("$btnName", 7.sp, borderColor, Colors.white, 4.8),
                  ],
                ),
              ),
            ),
          ),
          Positioned.fill(
            child: Material(
              color: Colors.transparent,
              child: InkWell(
                borderRadius: BorderRadius.circular(20),
                onTap: clickedFunc,
              ),
            ),
          ),
          IgnorePointer(
            child: Container(
              decoration: BoxDecoration(
                borderRadius: BorderRadius.circular(20),
                border: Border.all(color: borderColor, width: 3.5),
                color: Colors.transparent,
              ),
            ),
          ),
        ],
      ),
    );
  }
}

Widget modalBtn(BuildContext context, String text, Color backgroundColor,
    bool isTextblack) {
  // 모달 버튼
  return Container(
    width: 55.w,
    height: 60.h,
    alignment: Alignment.center,
    decoration: BoxDecoration(
      color: backgroundColor,
      borderRadius: BorderRadius.circular(15),
    ),
    child: Text(text,
        style: TextStyle(
            color: isTextblack ? Colors.black : Colors.white,
            fontSize: 6.sp,
            fontWeight: FontWeight.w500)),
  );
}

void openModal(
  BuildContext context,
  int idx,
) {
  // 모달 열기
  showDialog(
    context: context,
    builder: (context) => AlertDialog(
      alignment: Alignment.center,
      insetPadding: EdgeInsets.zero,
      contentPadding: EdgeInsets.only(top: 20.h, bottom: 10.h),
      backgroundColor: Colors.white,
      content: SizedBox(
        width: 120.w,
        height: 150.h,
        child: Column(
          mainAxisAlignment: MainAxisAlignment.spaceAround,
          mainAxisSize: MainAxisSize.min,
          children: [
            SizedBox(
              height: 15,
            ),
            Text(
              '다시 시작하시겠습니까?',
              textAlign: TextAlign.center,
              style: TextStyle(
                fontSize: 7.sp,
                color: Color(0xFF4A4A4A),
                fontWeight: FontWeight.w600,
              ),
            ),
            SizedBox(height: 20.h),
            Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                TextButton(
                  style: TextButton.styleFrom(
                    padding: EdgeInsets.zero,
                    minimumSize: Size(0, 0),
                  ),
                  onPressed: () {
                    Navigator.of(context).pop();
                  },
                  child: modalBtn(
                      context, '취소', Color.fromARGB(255, 205, 203, 202), true),
                ),
                SizedBox(width: 3.w),
                TextButton(
                  style: TextButton.styleFrom(
                    padding: EdgeInsets.zero,
                    minimumSize: Size(0, 0),
                  ),
                  onPressed: () {
                    // 다시하기 처리
                    Navigator.pushReplacement(
                      context,
                      MaterialPageRoute(
                        builder: (context) => PatternFillScreen(
                          index: idx,
                        ),
                      ),
                    );
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
