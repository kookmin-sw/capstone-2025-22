import 'package:flutter/material.dart';

class SignUpScreen extends StatefulWidget {
  const SignUpScreen({super.key});

  @override
  State<SignUpScreen> createState() => _SignUpScreenState();
}

void goToMain() {
  return;
}

class _SignUpScreenState extends State<SignUpScreen> {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Theme.of(context).scaffoldBackgroundColor,
      body: Padding(
        padding: const EdgeInsets.symmetric(
          horizontal: 30,
          vertical: 10,
        ),
        child: Column(
          children: [
            // 뒤로가기 버튼, 제목
            Row(
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
                        size: 40,
                        color: Colors.black,
                      ),
                    ),
                  ),
                ),
                // 제목: 화면의 flex 2 차지
                Expanded(
                  flex: 2,
                  child: Center(
                    child: Text(
                      "회원가입",
                      style: TextStyle(
                        fontSize: 30,
                        fontWeight: FontWeight.bold,
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
            ),

            Divider(
              height: 30,
              thickness: 1,
              color: Colors.grey[300],
            ),

            // 나머지 입력 폼
            Expanded(
              child: Center(
                child: Text(
                  "입력창들 들어갈 자리",
                  style: TextStyle(fontSize: 18, color: Colors.grey),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
