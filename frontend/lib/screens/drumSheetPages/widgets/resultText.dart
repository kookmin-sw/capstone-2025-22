import 'package:flutter/material.dart';
import 'package:capstone_2025/widgets/linedText.dart';
import 'package:flutter_screenutil/flutter_screenutil.dart';

class ResultText extends StatelessWidget {
  // 결과창 텍스트 (linedText + 그림자 효과)
  final String keyword;
  final Color txtColor;
  final Color borderColor;

  const ResultText({
    super.key,
    required this.keyword,
    required this.txtColor,
    required this.borderColor,
  });

  @override
  Widget build(BuildContext context) {
    return Stack(children: [
      Text(
        "$keyword",
        style: TextStyle(
          fontSize: (keyword == "PERFECT") ? 27.sp : 32.sp,
          fontWeight: FontWeight.bold,
          color: Colors.transparent,
          shadows: [
            Shadow(
              offset: Offset(4, 4),
              blurRadius: 35,
              color: Colors.black.withOpacity(0.7),
            ),
          ],
          height: 1,
        ),
      ),
      linedText(
        "$keyword",
        (keyword == "PERFECT") ? 27.sp : 32.sp,
        borderColor,
        txtColor,
        9.5,
      ),
    ]);
  }
}
