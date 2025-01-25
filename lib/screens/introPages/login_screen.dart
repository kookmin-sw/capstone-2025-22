import 'package:capstone_2025/screens/introPages/widgets/build_text_field.dart';
import 'package:flutter/material.dart';

class LoginScreen extends StatefulWidget {
  const LoginScreen({super.key});

  @override
  State<LoginScreen> createState() => _LoginScreenState();
}

class _LoginScreenState extends State<LoginScreen> {
  bool _isPasswordVisible = false; // 비밀번호 보기 상태 관리

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      // 백그라운드 컬러 context 이용해서 받아옴
      backgroundColor: Theme.of(context).scaffoldBackgroundColor,
      body: Center(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 20.0),
          child: SizedBox(
            width: 400, // 입력 칸과 버튼의 동일한 너비
            child: SingleChildScrollView(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                crossAxisAlignment: CrossAxisAlignment.center,
                children: [
                  SizedBox(height: 70),
                  Text(
                    '🥁알려드럼🥁',
                    style: TextStyle(
                      fontSize: 38.0,
                      fontWeight: FontWeight.w900,
                    ),
                  ),
                  SizedBox(height: 20),
                  buildTextField(
                      hint: '아이디(이메일)', obscureText: false, suffixIcon: null),
                  SizedBox(height: 10),
                  buildTextField(
                      hint: '비밀번호',
                      obscureText: !_isPasswordVisible,
                      suffixIcon: IconButton(
                        icon: Icon(
                          _isPasswordVisible
                              ? Icons.visibility
                              : Icons.visibility_off,
                        ),
                        onPressed: () {
                          setState(
                            () {
                              _isPasswordVisible =
                                  !_isPasswordVisible; // 비밀번호 보기 상태 변경
                            },
                          );
                        },
                      )),
                  SizedBox(height: 10),
                  ElevatedButton(
                    style: ElevatedButton.styleFrom(
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(12.0), // 버튼 테두리 둥글게
                      ),
                      padding: EdgeInsets.symmetric(vertical: 16.0), // 버튼 높이 설정
                      backgroundColor: Color(0xFF424242), // 버튼 배경색
                    ),
                    onPressed: () {
                      // 로그인 버튼 클릭
                    },
                    child: Center(
                      child: Text(
                        '로그인',
                        style: TextStyle(
                          fontSize: 15.0,
                          color: Colors.white,
                        ),
                      ),
                    ),
                  ),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.end,
                    children: [
                      TextButton(
                        onPressed: () {
                          // 비밀번호 찾기
                        },
                        child: Text('비밀번호 찾기', style: TextStyle(fontSize: 13)),
                      ),
                      Text('|'),
                      TextButton(
                        onPressed: () {
                          // 회원가입
                        },
                        child: Text('회원가입', style: TextStyle(fontSize: 13)),
                      ),
                    ],
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }

  // 입력 칸
}
