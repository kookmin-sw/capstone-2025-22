import 'package:flutter/material.dart';

Widget linedText(String text, double fontSize, Color borderColor,
    Color textColor, double strokeWidth) {
  return Stack(
    alignment: Alignment.topCenter, // 텍스트의 위쪽 여백을 없애기 위한 설정
    children: [
      Text(
        text,
        maxLines: 1, // 한 줄만 보여주고
        overflow: TextOverflow.ellipsis, // 뒤에 … 처리
        style: TextStyle(
          fontSize: fontSize,
          fontWeight: FontWeight.bold,
          foreground: Paint()
            ..style = PaintingStyle.stroke
            ..strokeWidth = strokeWidth
            ..color = borderColor,
          height: 1.0, // line height를 1로 설정하여 불필요한 여백을 제거
        ),
        textAlign: TextAlign.center,
      ),
      Text(
        text,
        maxLines: 1, // 한 줄만 보여주고
        overflow: TextOverflow.ellipsis, // 뒤에 … 처리
        style: TextStyle(
          fontSize: fontSize,
          fontWeight: FontWeight.bold,
          color: textColor,
          height: 1.0, // line height를 1로 설정하여 불필요한 여백을 제거
        ),
        textAlign: TextAlign.center,
      ),
    ],
  );
}
