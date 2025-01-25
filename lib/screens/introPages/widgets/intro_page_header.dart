import 'package:capstone_2025/screens/introPages/sign_up_screen.dart';
import 'package:flutter/material.dart';

class introPageHeader extends StatelessWidget {
  final String title;

  introPageHeader({required this.title});

  @override
  Widget build(BuildContext context) {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.center,
      children: [
        // 뒤로가기 버튼: 화면의 flex 1 차지
        Expanded(
          flex: 1,
          child: Align(
            alignment: Alignment.centerLeft,
            child: IconButton(
              onPressed: goToMain,
              icon: Icon(
                Icons.arrow_back_sharp,
                size: 65,
                color: Colors.black,
              ),
            ),
          ),
        ),
        // 제목: 화면의 flex 2 차지
        Expanded(
          flex: 1,
          child: Center(
            child: Text(
              title,
              style: TextStyle(
                fontSize: 30,
                fontWeight: FontWeight.w900,
              ),
            ),
          ),
        ),
        // 오른쪽 빈 공간: 화면의 flex 1 차지
        Expanded(
          flex: 1,
          child: SizedBox(), // 빈 공간
        ),
      ],
    );
  }
}
