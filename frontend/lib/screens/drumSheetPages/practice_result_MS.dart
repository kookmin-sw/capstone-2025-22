// ignore_for_file: file_names
import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:flutter_screenutil/flutter_screenutil.dart';

import 'package:capstone_2025/widgets/linedText.dart';
import 'package:capstone_2025/widgets/innerShadow.dart';
import 'package:capstone_2025/widgets/openSheetModal.dart';

import 'package:capstone_2025/screens/mainPages/navigation_screens.dart';
import 'package:capstone_2025/screens/drumSheetPages/drum_sheet_player.dart';
import 'package:capstone_2025/screens/drumSheetPages/widgets/resultText.dart';

class PracticeResultMS extends StatefulWidget {
  const PracticeResultMS({
    super.key,
    required this.sheetId,
    required this.musicTitle,
    required this.musicArtist,
    required this.score,
    required this.xmlDataString,
    required this.practiceInfo,
  });

  final int sheetId; // 악보 ID
  final String musicTitle; // 제목
  final String musicArtist; // 아티스트
  final int score; // 점수
  final String xmlDataString; // MusicXML
  final List<Map<String, dynamic>> practiceInfo; // 1차 채점 결과

  @override
  State<PracticeResultMS> createState() => _PracticeResultMSState();
}

class _PracticeResultMSState extends State<PracticeResultMS> {
  late int score = 80; // widget.score; // 점수
  late bool isPerfect = (score == 100); // 퍼펙트 여부
  late String musicTitle = widget.musicTitle; // 제목
  late String musicArtist = widget.musicArtist; // 아티스트

  Widget resultKeyword(int score) {
    String keyword;
    Color txtColor;
    Color borderColor;

    if (isPerfect) {
      keyword = "PERFECT";
      txtColor = Color(0xffFD9B8A);
      borderColor = Color(0xffB95D4C);
    } else if (score >= 80) {
      keyword = "GREAT";
      txtColor = Color(0xffFFE89B);
      borderColor = Color(0xffF1B45F);
    } else if (score >= 60) {
      keyword = "GOOD";
      txtColor = Color(0xffB6FF9B);
      borderColor = Color(0xff73CA5D);
    } else if (score >= 40) {
      keyword = "SOSO";
      txtColor = Color(0xff8ABCFD);
      borderColor = Color(0xff4C7FB9);
    } else {
      keyword = "BAD";
      txtColor = Color(0xffA2A2A2);
      borderColor = Color(0xff646464);
    }

    return ResultText(
        keyword: keyword, txtColor: txtColor, borderColor: borderColor);
  }

  Widget resultScore(int score) {
    Color txtColor;
    Color borderColor;
    if (isPerfect) {
      txtColor = Color(0xffFD9B8A);
      borderColor = Color(0xffB95D4C);
    } else if (score >= 60) {
      txtColor = Color(0xffFFE89B);
      borderColor = Color(0xffF1B45F);
    } else {
      txtColor = Color(0xffD9D9D9);
      borderColor = Color(0xff949494);
    }

    return linedText(
      '$score',
      35.sp,
      borderColor,
      txtColor,
      9.5,
    );
  }

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
                          padding: EdgeInsets.only(
                            // padding으로 사이즈 조절
                            top: MediaQuery.of(context).size.height * 0.01,
                            bottom: MediaQuery.of(context).size.height * 0.1,
                            left: isPerfect
                                ? MediaQuery.of(context).size.width * 0.1
                                : MediaQuery.of(context).size.width * 0.1,
                            right: isPerfect
                                ? MediaQuery.of(context).size.width * 0.1
                                : MediaQuery.of(context).size.width * 0.1,
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
                                  padding: EdgeInsets.only(
                                      top: MediaQuery.of(context).size.height *
                                          0.03,
                                      bottom:
                                          MediaQuery.of(context).size.height *
                                              0.05,
                                      left: (isPerfect)
                                          ? MediaQuery.of(context).size.width *
                                              0.03
                                          : 20.w,
                                      right: (isPerfect)
                                          ? MediaQuery.of(context).size.width *
                                              0.03
                                          : 20.w),
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
                                                left: isPerfect
                                                    ? MediaQuery.of(context)
                                                            .size
                                                            .width *
                                                        0.03
                                                    : 10.w),
                                            child: Column(
                                              // 성공 여부 텍스트
                                              children: [
                                                SizedBox(
                                                  child: linedText(
                                                      "$musicTitle",
                                                      12.sp,
                                                      Colors.black
                                                          .withOpacity(0.3),
                                                      Colors.white,
                                                      7),
                                                ),
                                                resultKeyword(score),
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
                                                  horizontal: ((isPerfect)
                                                      ? 20.w
                                                      : 35.w)),
                                              child: Column(children: [
                                                linedText(
                                                  'SCORE',
                                                  10.sp,
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
                                                  resultScore(score),
                                                ])
                                              ]),
                                            ),
                                          ),
                                        ],
                                      ),
                                      SizedBox(height: 15.h),
                                      ButtonForm(
                                        btnName: "상세 기록 확인하기",
                                        buttonColor: Color(0xff949494),
                                        borderColor:
                                            Color.fromARGB(255, 104, 104, 104),
                                        shadowColor:
                                            Color.fromARGB(255, 177, 177, 177),
                                        width:
                                            MediaQuery.of(context).size.width *
                                                0.75,
                                        clickedFunc: () {
                                          openMusicSheet(
                                            context: context,
                                            xmlDataString: widget.xmlDataString,
                                            practiceInfo: widget.practiceInfo,
                                            isPatternMode: false,
                                          );
                                        },
                                        btnIcon:
                                            Icons.insert_drive_file_rounded,
                                      ),
                                      SizedBox(height: 20.h),
                                      Row(
                                        children: [
                                          Expanded(
                                            flex: 1,
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
                                                  openModal(
                                                      context,
                                                      widget.sheetId,
                                                      musicTitle,
                                                      musicArtist,
                                                      widget.xmlDataString);
                                                },
                                              ),
                                            ),
                                          ),
                                          SizedBox(width: 5.w),
                                          Expanded(
                                            flex: 1,
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
                                                        firstSelectedIndex: 3,
                                                      ),
                                                    ),
                                                  );
                                                },
                                              ),
                                            ),
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
  const ButtonForm({
    super.key,
    required this.btnName,
    required this.buttonColor,
    required this.borderColor,
    required this.clickedFunc,
    required this.shadowColor,
    this.width = double.infinity,
    this.btnIcon,
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
      height: MediaQuery.of(context).size.height * 0.15,
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
              padding: EdgeInsets.symmetric(horizontal: 16.w),
              child: Center(
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    if (btnIcon != null)
                      Icon(
                        btnIcon,
                        size: 11.sp,
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
    width: MediaQuery.of(context).size.width * 0.15,
    height: MediaQuery.of(context).size.height * 0.15,
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
  int sheetId,
  String musicTitle,
  String musicArtist,
  String xmlDataString,
) {
  // 모달 열기
  showDialog(
    context: context,
    builder: (context) => AlertDialog(
      alignment: Alignment.center,
      insetPadding: EdgeInsets.zero,
      contentPadding: EdgeInsets.symmetric(
          horizontal: MediaQuery.of(context).size.height * 0.02,
          vertical: MediaQuery.of(context).size.height * 0.03),
      backgroundColor: Colors.white,
      content: SizedBox(
        width: MediaQuery.of(context).size.width * 0.32,
        height: MediaQuery.of(context).size.height * 0.34,
        child: Column(
          mainAxisAlignment: MainAxisAlignment.spaceAround,
          mainAxisSize: MainAxisSize.min,
          children: [
            SizedBox(
              height: 15.h,
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
            SizedBox(height: 25.h),
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
                SizedBox(width: 5.w),
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
                        builder: (context) => DrumSheetPlayer(
                          sheetId: sheetId,
                          title: musicTitle,
                          artist: musicArtist,
                          sheetXmlData:
                              base64Encode(utf8.encode(xmlDataString)),
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
