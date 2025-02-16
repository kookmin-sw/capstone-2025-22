import 'dart:convert';
import 'package:capstone_2025/screens/introPages/find_pw_screen.dart';
import 'package:capstone_2025/screens/introPages/login_screen_google.dart';
import 'package:capstone_2025/screens/introPages/sign_up_screen.dart';
import 'package:capstone_2025/screens/introPages/widgets/build_text_field.dart';
import 'package:capstone_2025/screens/introPages/widgets/intro_page_header.dart';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:flutter_secure_storage/flutter_secure_storage.dart';

class LoginScreen extends StatefulWidget {
  const LoginScreen({super.key});

  @override
  State<LoginScreen> createState() => _LoginScreenState();
}

class _LoginScreenState extends State<LoginScreen> {
  final TextEditingController _emailController = TextEditingController();
  final TextEditingController _passwordController = TextEditingController();
  final _storage =
      const FlutterSecureStorage(); // 로그인 성공 시 받은 JWT 토큰을 저장. 이후 자동 로그인 기능 구현할 때 사용.
  bool _isPasswordVisible = false; // 비밀번호 보기 상태 관리
  bool _isLoading = false; // 로딩 상태 관리

  Future<void> _login() async {
    // 로그인 버튼 클릭 시
    final String email = _emailController.text.trim(); // 사용자가 입력한 아이디 가져오기
    final String password =
        _passwordController.text.trim(); // 사용자가 입력한 비밀번호 가져오기

    if (email.isEmpty || password.isEmpty) {
      // 아이디, 비밀번호 둘 중 하나라도 입력하지 않고 로그인 버튼을 눌렀을 경우 오류 메시지 출력
      _showSnackbar('아이디와 비밀번호를 입력하세요.');
      return;
    }

    setState(() {
      _isLoading = true; // 로딩 상태를 활성화 -> 로딩스피너 표시
    });

    try {
      // 서버에 로그인 요청
      final response = await http.post(
        // Uri.parse(
        //     'http://192.168.219.108:28080/auth/signin'), // API URL 수정해야 함!
        Uri.parse('http://10.0.2.2:28080/auth/signin'), // API URL 수정해야 함!
        headers: {'Content-Type': 'application/json'}, // 요청을 JSON 형식으로 보냄
        body: jsonEncode({'email': email, 'password': password}),
      );

      final data = jsonDecode(response.body);

      if (data['body'] == null) {
        // 로그인 실패
        _showSnackbar('로그인에 실패했습니다');
        print("실패");
      } else {
        // 로그인 성공
        // JWT 저장
        await _storage.write(
            key: 'access_token', value: data['body']['access_token']);
        await _storage.write(
            key: 'refresh_token', value: data['body']['refresh_token']);

        // 로그인 성공 시 화면으로 이동
        if (mounted) {
          Navigator.pushReplacement(
            context,
            MaterialPageRoute(
              builder: (context) => const LoginScreen(), // MainScreen으로 바꾸기!
            ),
          );
        }
      }
    } catch (e) {
      // 인터넷 연결 문제 또는 서버 오류 발생 시
      _showSnackbar('네트워크 오류: $e');
      print(e);
    } finally {
      setState(() {
        _isLoading = false; // 로딩 상태 해제 & 로그인 버튼 다시 활성화
      });
    }
  }

  void _showSnackbar(String message) {
    // 메시지 출력하는 함수
    ScaffoldMessenger.of(context)
        .showSnackBar(SnackBar(content: Text(message)));
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Theme.of(context).scaffoldBackgroundColor,
      body: Center(
        // 화면 중앙에 위젯을 배치
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 20.0),
          child: SingleChildScrollView(
            // 스크롤 가능
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              crossAxisAlignment: CrossAxisAlignment.center,
              children: [
                introPageHeader(
                  // 수정하기!!!
                  title: '🥁알려드럼🥁',
                  targetPage: LoginScreenGoogle(),
                ),
                const SizedBox(height: 20),
                SizedBox(
                  width: 400,
                  child: Column(
                    children: [
                      buildTextField(
                        // 아이디 입력 필드
                        controller: _emailController,
                        hint: '아이디(이메일)',
                        obscureText: false, // 가려지지 않음
                        suffixIcon: null,
                      ),
                      const SizedBox(height: 10),
                      buildTextField(
                        // 비밀번호 입력 필드
                        controller: _passwordController,
                        hint: '비밀번호',
                        obscureText: !_isPasswordVisible, // 비밀번호 보이기/숨기기 기능 활성화
                        suffixIcon: IconButton(
                          // 눈 모양 아이콘 클릭하면 비밀번호 보이게 함
                          icon: Icon(_isPasswordVisible
                              ? Icons.visibility
                              : Icons.visibility_off),
                          onPressed: () {
                            setState(() {
                              // 눈 모양 아이콘 클릭할 때마다 상태 변경
                              _isPasswordVisible = !_isPasswordVisible;
                            });
                          },
                        ),
                      ),
                      const SizedBox(height: 10),
                      _isLoading
                          ? const CircularProgressIndicator() // 로딩 중이면 로딩스피너 표시
                          : ElevatedButton(
                              // _isLoading이 false이면 로그인 버튼 활성화
                              style: ElevatedButton.styleFrom(
                                shape: RoundedRectangleBorder(
                                  borderRadius: BorderRadius.circular(12.0),
                                ),
                                padding:
                                    const EdgeInsets.symmetric(vertical: 16.0),
                                backgroundColor: Color(0xFF424242),
                              ),
                              onPressed: _login, // 로그인 버튼 클릭하면 _login 함수 호출
                              child: const Center(
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
                        mainAxisAlignment: MainAxisAlignment.end, // 오른쪽 끝 정렬
                        children: [
                          TextButton(
                            // '비밀번호 찾기' 버튼
                            onPressed: () {
                              Navigator.push(
                                context,
                                MaterialPageRoute(
                                  builder: (context) =>
                                      const FindPwScreen(), // 버튼 클릭하면 FindPwScreen으로 이동
                                ),
                              );
                            },
                            child: const Text('비밀번호 찾기',
                                style: TextStyle(fontSize: 13)),
                          ),
                          const Text('|'),
                          TextButton(
                            // '회원가입' 버튼
                            onPressed: () {
                              Navigator.push(
                                context,
                                MaterialPageRoute(
                                  builder: (context) =>
                                      const SignUpScreen(), // 버튼 클릭하면 SignUpScreen으로 이동
                                ),
                              );
                            },
                            child: const Text('회원가입',
                                style: TextStyle(fontSize: 13)),
                          ),
                        ],
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
