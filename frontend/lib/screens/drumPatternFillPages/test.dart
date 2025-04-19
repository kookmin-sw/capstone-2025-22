// 테스트 페이지야. 패턴 및 필인 연습 목록 페이지 만들고 이거 삭제해줘.

import 'package:flutter/material.dart';
import 'package:capstone_2025/screens/drumPatternFillPages/pattern_fill_screen.dart';

class TestPage extends StatelessWidget {
  const TestPage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("테스트 페이지")),
      body: Center(
        child: ElevatedButton(
          onPressed: () {
            Navigator.push(
              context,
              MaterialPageRoute(builder: (_) => const PatternFillScreen()),
            );
          },
          child: const Text("패턴 및 필인 연습"),
        ),
      ),
    );
  }
}
