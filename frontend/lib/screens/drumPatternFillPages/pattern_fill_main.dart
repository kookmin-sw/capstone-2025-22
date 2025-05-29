import 'package:capstone_2025/services/api_func.dart';
import 'package:capstone_2025/services/storage_service.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:capstone_2025/widgets/innerShadow.dart';
import 'package:capstone_2025/widgets/linedText.dart';
import 'package:capstone_2025/screens/drumPatternFillPages/pattern_fill_screen.dart';
import 'package:flutter_screenutil/flutter_screenutil.dart';

class PatternFillMain extends StatefulWidget {
  const PatternFillMain({super.key});

  @override
  State<PatternFillMain> createState() => _PatternFillMainState();
}

class _PatternFillMainState extends State<PatternFillMain> {
  bool _isLoading = true;
  List<Widget> _patternWidgets = [];

  @override
  void initState() {
    super.initState();
    loadPatternList();
  }

  Future<void> loadPatternList() async {
    _patternWidgets = await buildPatternList();
    if (!mounted) return;
    setState(() {
      _isLoading = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: _isLoading
          ? const Center(child: CircularProgressIndicator())
          : Column(
              children: [
                Expanded(
                  child: Padding(
                    padding: EdgeInsets.all(25.h),
                    child: Column(
                      children: [
                        Expanded(
                          flex: 3,
                          child: Center(
                            child: linedText("LEVEL", 12.5.sp, Colors.black45,
                                Colors.white, 5.5),
                          ),
                        ),
                        Spacer(flex: 1),
                        Expanded(
                          flex: 20,
                          child: SingleChildScrollView(
                            physics: ClampingScrollPhysics(),
                            child: Column(children: _patternWidgets),
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ],
            ),
    );
  }

  Widget clickedListItem(BuildContext context, int index, bool isLevelCleared,
      bool isLevelLocked, int score) {
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
        score,
      ),
    );
  }

  Widget modalBtn(BuildContext context, String text, Color backgroundColor,
      bool isTextblack) {
    // 모달 버튼
    return Container(
      width: 50.w,
      height: 60.h,
      alignment: Alignment.center,
      decoration: BoxDecoration(
        color: backgroundColor,
        borderRadius: BorderRadius.circular(15),
      ),
      child: Text(text,
          style: TextStyle(
              color: isTextblack ? Colors.black : Colors.white,
              fontSize: 5.5.sp,
              fontWeight: FontWeight.w500)),
    );
  }

  void openModal(BuildContext context, int index) {
    showDialog(
      context: context,
      builder: (context) => Dialog(
        backgroundColor: Colors.transparent,
        insetPadding: EdgeInsets.zero,
        child: Container(
          width: MediaQuery.of(context).size.width * 0.3,
          margin: EdgeInsets.symmetric(horizontal: 0.w),
          padding: EdgeInsets.all(20.h),
          decoration: BoxDecoration(
            color: Colors.white,
            borderRadius: BorderRadius.circular(16),
          ),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Text(
                'Basic Pattern $index',
                textAlign: TextAlign.center,
                style: TextStyle(
                  fontSize: 7.5.sp,
                  fontWeight: FontWeight.bold,
                  color: Color(0xFF4A4A4A),
                ),
              ),
              SizedBox(height: 10.h),
              Text(
                '연습을 시작하시겠습니까?',
                textAlign: TextAlign.center,
                style: TextStyle(
                  fontSize: 6.5.sp,
                  color: Color(0xFF4A4A4A),
                ),
              ),
              SizedBox(height: 20.h),
              Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Flexible(
                    child: SizedBox(
                      height: MediaQuery.of(context).size.height * 0.11,
                      width: MediaQuery.of(context).size.width * 0.15,
                      child: ElevatedButton(
                        style: ElevatedButton.styleFrom(
                          padding: EdgeInsets.zero,
                          minimumSize: Size(0, 0),
                          backgroundColor: Color(0xFFF2F2F2),
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(12),
                          ),
                        ),
                        onPressed: () {
                          Navigator.of(context).pop();
                        },
                        child: Text(
                          '취소',
                          style: TextStyle(
                            color: Color(0xFF646464),
                            fontSize: 5.5.sp,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                      ),
                    ),
                  ),
                  SizedBox(width: 5.w),
                  Flexible(
                    child: SizedBox(
                      height: MediaQuery.of(context).size.height * 0.11,
                      width: MediaQuery.of(context).size.width * 0.15,
                      child: ElevatedButton(
                        style: ElevatedButton.styleFrom(
                          padding: EdgeInsets.zero,
                          minimumSize: Size(0, 0),
                          backgroundColor: Color(0xffD97D6C),
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(12),
                          ),
                        ),
                        onPressed: () {
                          Navigator.of(context).pop();
                          Navigator.of(context).push(
                            MaterialPageRoute(
                              builder: (context) =>
                                  PatternFillScreen(index: index),
                            ),
                          );
                        },
                        child: Text(
                          '확인',
                          style: TextStyle(
                            color: Colors.white,
                            fontSize: 5.5.sp,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                      ),
                    ),
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
      bool isLevelLocked, int score) {
    double containerHeight = 65.h;
    double containerWidth = MediaQuery.of(context).size.width * 0.67;
    double borderRadius = 13;
    double fontSize = 8.sp;

    return Padding(
      padding: EdgeInsets.only(bottom: 10.h),
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
              padding: EdgeInsets.all(5.0.h),
              child: Row(
                // List item 내부 요소 - 텍스트 점수 등
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Padding(
                    padding: EdgeInsets.only(left: 13.w),
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
                    padding: EdgeInsets.only(right: 13.w),
                    child: SizedBox(
                      width: 20.w,
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
                                  size: 12.sp,
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

  Future<List<Widget>> buildPatternList() async {
    String? email = await storage.read(key: "user_email");
    List<Widget> levels = [];

    Map<int, int> patternScores = {}; // patternId -> score

    var successPatterns = await getHTTP('/patterns/success', {"email": email});
    if (successPatterns['errMessage'] == null) {
      var body = successPatterns['body'] as List;
      for (var entry in body) {
        patternScores[entry['patternId'] as int] = entry['score'] as int;
      }
    } else {
      print(successPatterns['errMessage']);
    }

    var patterns = await getHTTP('/patterns', {}, reqHeader: {});
    if (patterns['errMessage'] == null) {
      var body = patterns['body'];
      levels = (body as List).asMap().entries.map((entry) {
        int index = entry.key + 1;
        bool isCleared = patternScores.containsKey(index);
        bool isLocked = !patternScores.containsKey(index - 1) && index != 1;
        int score = patternScores[index] ?? 0;

        return mounted
            ? clickedListItem(context, index, isCleared, isLocked, score)
            : const SizedBox.shrink();
      }).toList();
    } else {
      print(patterns['errMessage']);
    }

    return levels;
  }
}
