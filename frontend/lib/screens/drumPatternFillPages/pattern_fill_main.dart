import 'package:capstone_2025/services/api_func.dart';
import 'package:capstone_2025/services/storage_service.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:capstone_2025/widgets/innerShadow.dart';
import 'package:capstone_2025/widgets/linedText.dart';
import 'package:capstone_2025/screens/drumPatternFillPages/pattern_fill_screen.dart';

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
                    padding: const EdgeInsets.all(20),
                    child: Column(
                      children: [
                        Expanded(
                          flex: 3,
                          child: Center(
                            child: linedText(
                                "LEVEL", 35, Colors.black45, Colors.white, 5.5),
                          ),
                        ),
                        Spacer(flex: 1),
                        Expanded(
                          flex: 20,
                          child: SingleChildScrollView(
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
      width: 150,
      // MediaQuery.of(context).size.width * 0.168,
      height: 53,
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
        backgroundColor: Colors.white,
        content: SizedBox(
          width: 315,
          height: 140,
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Padding(
                padding: const EdgeInsets.only(top: 10, bottom: 0),
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
                    child: modalBtn(context, '취소',
                        Color.fromARGB(255, 205, 203, 202), true),
                  ),
                  SizedBox(width: 15),
                  TextButton(
                    style: TextButton.styleFrom(
                      padding: EdgeInsets.zero,
                      minimumSize: Size(0, 0),
                    ),
                    onPressed: () {
                      Navigator.of(context).pop();
                      Navigator.of(context).push(
                        MaterialPageRoute(
                          builder: (context) => PatternFillScreen(index: index),
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

  Future<List<Widget>> buildPatternList() async {
    String? email = await storage.read(key: "user_email");
    List<Widget> levels = [];
    int lastPatternId = -1;

    var successPatterns = await getHTTP('/patterns/success', {"email": email});
    if (successPatterns['errMessage'] == null) {
      var body = successPatterns['body'] as List;
      lastPatternId = body.isNotEmpty ? body.last['patternId'] as int : 0;
    } else {
      print(successPatterns['errMessage']);
    }

    var patterns = await getHTTP('/patterns', {}, reqHeader: {});
    if (patterns['errMessage'] == null) {
      var body = patterns['body'];
      levels = (body as List).asMap().entries.map((entry) {
        int index = entry.key + 1;
        bool isCleared = (lastPatternId > index);
        bool isLocked = (lastPatternId + 1 < index);

        return mounted
            ? clickedListItem(context, index, isCleared, isLocked)
            : const SizedBox.shrink();
      }).toList();
    } else {
      print(patterns['errMessage']);
    }

    return levels;
  }
}
