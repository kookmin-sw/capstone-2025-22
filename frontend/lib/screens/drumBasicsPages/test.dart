// 팝업창 테스트하려고 만든 파일이야. 드럼 기초 페이지 만든 후 이 파일 삭제해줘!

import 'package:flutter/material.dart';
import 'drum_info_popup.dart'; // 위젯 파일 import

class Test extends StatelessWidget {
  const Test({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: ElevatedButton(
          onPressed: () {
            showDialog(
              context: context,
              builder: (context) => const DrumInfoPopup(
                title: "드럼 종류", // 팝업 이름
                imagePath: "assets/images/drum_kit.jpg", // 이미지
              ),
            );
          },
          child: const Text("팝업 버튼"),
        ),
      ),
    );
  }
}
