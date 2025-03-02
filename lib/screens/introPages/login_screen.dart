import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import 'package:capstone_2025/screens/mainPages/my_page.dart';
import 'package:capstone_2025/screens/introPages/sign_up_screen.dart';
import 'package:capstone_2025/screens/introPages/find_pw_screen.dart';
import 'package:capstone_2025/screens/introPages/login_screen_google.dart';
import 'package:capstone_2025/screens/introPages/widgets/build_text_field.dart';
import 'package:capstone_2025/screens/introPages/widgets/intro_page_header.dart';

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

  // 로그인 버튼 클릭 시
  Future<void> _login() async {
    final String email = _emailController.text.trim(); // 사용자가 입력한 아이디 가져오기
    final String password =
        _passwordController.text.trim(); // 사용자가 입력한 비밀번호 가져오기

    // 예외처리1: 정보가 누락되었을 때
    if (email.isEmpty || password.isEmpty) {
      _showSnackbar('아이디와 비밀번호를 입력하세요.');
      return;
    }

    // 예외처리2: 이메일 주소 형식이 잘못되었을 때
    final emailRegex = RegExp(
        r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'); // 이메일 형식 확인
    final validateEmail = emailRegex.hasMatch(email);
    if (!validateEmail) {
      _showSnackbar("이메일 주소 형식이 잘못됐습니다.");
      return;
    }

    setState(() {
      _isLoading = true; // 로딩스피너 표시
    });

    try {
      // 서버에 로그인 요청
      final response = await http.post(
        Uri.parse('http://10.0.2.2:28080/auth/signin'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'email': email, 'password': password}),
      );

      final data = jsonDecode(response.body);

      print("response.statusCode: ${response.statusCode}"); // 에러 코드 확인

      // 로그인 성공!
      if (response.statusCode == 200 && data['body'] != null) {
        // ignore: unused_local_variable
        final String userEmail = data['body']['email'];
        final String userName = data['body']['name'];
        final String accessToken = data['body']['access_token'];
        final String refreshToken = data['body']['refresh_token'];

        // secure storage에 저장
        await _storage.write(key: 'user_email', value: userEmail);
        await _storage.write(key: 'user_name', value: userName);
        await _storage.write(key: 'access_token', value: accessToken);
        await _storage.write(key: 'refresh_token', value: refreshToken);

        _showSnackbar('$userName님 환영합니다.');

        // 로그인 성공 시 메인 화면으로 이동
        if (mounted) {
          Navigator.pushReplacement(
            context,
            MaterialPageRoute(
              builder: (context) => const MyPage(),
            ),
          );
        }
      } else {
        // 로그인 실패
        _handleError(response.statusCode);
      }
    } catch (e) {
      // 인터넷 연결 문제 또는 서버 오류 발생 시
      _showSnackbar('네트워크 오류: $e');
    } finally {
      setState(() {
        _isLoading = false; // 로딩 상태 해제 & 로그인 버튼 다시 활성화
      });
    }
  }

  // 에러 코드에 따른 에러 처리
  void _handleError(int statusCode) {
    switch (statusCode) {
      case 400: // 정보가 누락되었을 때
        _showSnackbar('아이디와 비밀번호를 입력하세요.');
        break;
      case 403: // 권한이 없을 때
        _showSnackbar('접근 권한이 없습니다.');
        break;
      case 404: // 사용자 정보가 없을 때
        _showSnackbar('사용자 정보를 찾을 수 없습니다.');
        break;
      case 500: // 서버 내부 오류
        _showSnackbar('서버 오류가 발생했습니다. 잠시 후 다시 시도하세요.');
        break;
      default:
        _showSnackbar('알 수 없는 오류가 발생했습니다.');
        break;
    }
  }

  // 페이지 하단에 메시지 출력하는 함수
  void _showSnackbar(String message) {
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
