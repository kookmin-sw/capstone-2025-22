// 화면 크기에 맞춰 자동으로 레이아웃을 조절하는 공통 위젯 추가
import 'package:flutter/material.dart';

class ResponsiveScaffold extends StatelessWidget {
  final double maxWidth, maxHeight;
  final Orientation orientation;
  final Widget body;

  const ResponsiveScaffold({
    super.key,
    required this.maxWidth,
    required this.maxHeight,
    required this.orientation,
    required this.body,
  });

  @override
  Widget build(BuildContext context) {
    if (orientation == Orientation.landscape) {
      return Row(children: [
        Expanded(child: body),
        Container(width: maxWidth * 0.2, color: Colors.grey[200]),
      ]);
    }
    return Scaffold(body: body);
  }
}
