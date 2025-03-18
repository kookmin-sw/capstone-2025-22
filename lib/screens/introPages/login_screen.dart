import 'dart:convert'; // JSON 변환을 위한 패키지
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http; // HTTP 요청을 위한 패키지
import 'package:flutter_secure_storage/flutter_secure_storage.dart'; // secure storage
import 'package:capstone_2025/screens/introPages/sign_up_screen.dart';
import 'package:capstone_2025/screens/introPages/find_pw_screen.dart';
import 'package:capstone_2025/screens/mainPages/navigation_screens.dart';
import 'package:capstone_2025/screens/introPages/login_screen_google.dart';
import 'package:capstone_2025/screens/introPages/widgets/build_text_field.dart';
import 'package:capstone_2025/screens/introPages/widgets/intro_page_header.dart';
import 'package:capstone_2025/screens/mainPages/navigation_screens.dart';

/// 일반 로그인 화면
class LoginScreen extends StatefulWidget {
  const LoginScreen({super.key});

  @override
  State<LoginScreen> createState() => _LoginScreenState();
}

class _LoginScreenState extends State<LoginScreen> {
  late final TextEditingController _emailController; // 이메일 입력 필드 컨트롤러
  late final TextEditingController _passwordController; // 비밀번호 입력 필드 컨트롤러
  final FlutterSecureStorage _storage =
      const FlutterSecureStorage(); // 로그인 성공 시 받은 JWT 토큰을 저장. 이후 자동 로그인 기능 구현할 때 사용

  bool _isPasswordVisible = false; // 비밀번호 보기&숨기기 상태
  bool _isLoading = false; // 로딩 상태
  String? _errorMessage; // 오류 메시지

  @override
  void initState() {
    super.initState();
    _emailController = TextEditingController(); // 이메일 입력 필드 초기화
    _passwordController = TextEditingController(); // 비밀번호 입력 필드 초기화
  }

  @override
  void dispose() {
    // 메모리 누수 방지를 위해 컨트롤러 해제
    _emailController.dispose();
    _passwordController.dispose();
    super.dispose();
  }

  // 로그인 버튼 클릭 시 실행되는 함수
  Future<void> _login() async {
    final String email = _emailController.text.trim(); // 사용자가 입력한 이메일 값 가져오기
    final String password =
        _passwordController.text.trim(); // 사용자가 입력한 비밀번호 값 가져오기

    setState(() => _errorMessage = null); // 기존 오류 메시지 초기화

    // 예외처리1: 입력 필드가 비어있는지 검사
    if (email.isEmpty || password.isEmpty) {
      setState(() => _errorMessage = '아이디 또는 비밀번호를 확인해주세요.');
      return;
    }

    // 예외처리2: 이메일 형식이 올바른지 정규식으로 검사
    final emailRegex =
        RegExp(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$');
    if (!emailRegex.hasMatch(email)) {
      setState(() => _errorMessage = '이메일 주소 형식이 잘못됐습니다.');
      return;
    }

    setState(() => _isLoading = true); // 로딩스피너 표시

    try {
      // 서버에 로그인 요청 보내기
      final response = await http.post(
        Uri.parse('http://10.0.2.2:28080/auth/signin'),
        headers: {
          "Content-Type": "application/json",
          "Accept": "application/json",
        },
        body: jsonEncode({'email': email, 'password': password}),
      );

      print("response.statusCode: ${response.statusCode}"); // 에러 코드 확인
      print(response.body);

      if (response.statusCode == 200) {
        // 로그인 성공 시
        final data = jsonDecode(response.body);
        await _saveUserData(data['body']); // Secure Storage에 사용자 정보 저장

        // 페이지 하단에 환영 메시지 출력
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(content: Text('${data['body']['nickname']}님 환영합니다.')),
          );
        }

        // 메인 화면으로 이동
        if (mounted) {
          Navigator.pushReplacement(
            context,
            MaterialPageRoute(builder: (context) => NavigationScreens()),
          );
        }
      } else {
        // 로그인 실패 시 에러 코드 처리
        _handleError(response.statusCode);
      }
    } catch (e) {
      setState(() => _errorMessage = '네트워크 오류가 발생했습니다.');
      print(e);
    } finally {
      setState(() => _isLoading = false); // 로딩스피너 해제
    }
  }

  /// 로그인 성공 시 사용자 정보 저장
  Future<void> _saveUserData(Map<String, dynamic> userData) async {
    await _storage.deleteAll(); // 기존 데이터 초기화

    await _storage.write(key: 'user_email', value: userData['email']);
    await _storage.write(key: 'nick_name', value: userData['nickname']);
    await _storage.write(key: 'access_token', value: userData['accessToken']);
    await _storage.write(key: 'refresh_token', value: userData['refreshToken']);
  }

  /// 에러 코드 처리
  void _handleError(int statusCode) {
    final Map<int, String> errorMessages = {
      400: '아이디와 비밀번호를 입력하세요.',
      403: '접근 권한이 없습니다.',
      404: '사용자 정보를 찾을 수 없습니다.',
      500: '서버 오류가 발생했습니다. 잠시 후 다시 시도하세요.',
    };
    setState(() =>
        _errorMessage = (errorMessages[statusCode] ?? '알 수 없는 오류가 발생했습니다.'));
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Theme.of(context).scaffoldBackgroundColor,
      body: Center(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 20.0),
          // 스크롤 가능
          child: SingleChildScrollView(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              crossAxisAlignment: CrossAxisAlignment.center,
              children: [
                introPageHeader(
                    title: '🥁알려드럼🥁', targetPage: LoginScreenGoogle()),
                const SizedBox(height: 20),
                SizedBox(
                  width: 400, // 입력 필드의 최대 너비 설정
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
                        obscureText: !_isPasswordVisible, // 비밀번호 보기 상태 기능 활성화
                        suffixIcon: IconButton(
                          // 눈 모양 아이콘 클릭하면 비밀번호 보이게 함
                          icon: Icon(_isPasswordVisible
                              ? Icons.visibility
                              : Icons.visibility_off),
                          onPressed: () {
                            // 아이콘 클릭할 때마다 상태 변경
                            setState(
                                () => _isPasswordVisible = !_isPasswordVisible);
                          },
                        ),
                      ),
                      if (_errorMessage != null) _buildErrorMessage(),
                      const SizedBox(height: 10),
                      _isLoading
                          ? const CircularProgressIndicator() // 로딩 중이면 로딩스피너 표시
                          : _buildLoginButton(), // _isLoading이 false이면 로그인 버튼 활성화
                      _buildBottomLinks(),
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

  /// 오류 메시지
  Widget _buildErrorMessage() {
    return Align(
      alignment: Alignment.centerLeft, // 왼쪽 정렬
      child: Padding(
        padding: const EdgeInsets.only(left: 8.0),
        child: Text(
          _errorMessage!,
          style: TextStyle(color: Colors.red),
        ),
      ),
    );
  }

  /// 로그인 버튼
  Widget _buildLoginButton() {
    return ElevatedButton(
      style: ElevatedButton.styleFrom(
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(12.0),
        ),
        padding: const EdgeInsets.symmetric(vertical: 16.0),
        backgroundColor: Color(0xFF424242),
      ),
      onPressed: _login, // 로그인 버튼 클릭하면 _login 함수 호출
      child: const Center(
        child: Text(
          '로그인',
          style: TextStyle(fontSize: 15.0, color: Colors.white),
        ),
      ),
    );
  }

// 하단 '비밀번호 찾기 | 회원가입' 링크
  Widget _buildBottomLinks() {
    return Row(
      mainAxisAlignment: MainAxisAlignment.end, // 오른쪽 정렬
      children: [
        TextButton(
          // '비밀번호 찾기' 버튼
          onPressed: () => Navigator.push(
              context, MaterialPageRoute(builder: (_) => const FindPwScreen())),
          child: const Text('비밀번호 찾기', style: TextStyle(fontSize: 13)),
        ),
        const Text('|'),
        TextButton(
          // '회원가입' 버튼
          onPressed: () => Navigator.push(
              context, MaterialPageRoute(builder: (_) => const SignUpScreen())),
          child: const Text('회원가입', style: TextStyle(fontSize: 13)),
        ),
      ],
    );
  }
}
