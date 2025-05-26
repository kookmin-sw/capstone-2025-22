import 'dart:convert'; // JSON 변환을 위한 패키지
import 'package:capstone_2025/services/storage_service.dart';
import 'package:flutter/material.dart';
import 'package:capstone_2025/services/api_func.dart';
import 'package:capstone_2025/screens/introPages/sign_up_screen.dart';
import 'package:capstone_2025/screens/introPages/find_pw_screen.dart';
import 'package:capstone_2025/screens/mainPages/navigation_screens.dart';
import 'package:capstone_2025/screens/introPages/login_screen_google.dart';
import 'package:capstone_2025/screens/introPages/widgets/build_text_field.dart';
import 'package:flutter_screenutil/flutter_screenutil.dart';

/// 일반 로그인 화면
class LoginScreen extends StatefulWidget {
  const LoginScreen({super.key});

  @override
  State<LoginScreen> createState() => _LoginScreenState();
}

class _LoginScreenState extends State<LoginScreen> {
  late final TextEditingController _emailController; // 이메일 입력 필드 컨트롤러
  late final TextEditingController _passwordController; // 비밀번호 입력 필드 컨트롤러

  bool _isPasswordVisible = false; // 비밀번호 보기&숨기기 상태
  bool _isLoading = false; // 로딩 상태
  String? _errorMessage; // 오류 메시지

  bool get _keyboardIsVisible => MediaQuery.of(context).viewInsets.bottom != 0;

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
      final Map<String, dynamic> requestBody = {
        // HTTP 함수를 통해 보낼 request body
        'email': email,
        'password': password,
      };

      final userInfo = await postHTTP("/auth/signin", requestBody);

      if (userInfo['errMessage'] == null) {
        // 로그인 성공 시
        await saveUserData(userInfo); // Secure Storage에 사용자 정보 저장

        // 페이지 하단에 환영 메시지 출력
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(content: Text('${userInfo['body']['nickname']}님 환영합니다.')),
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
        print("로그인 실패: ${userInfo['errMessage']}");
        setState(() => _errorMessage = ("로그인 실패: ${userInfo['errMessage']}"));
      }
    } catch (error) {
      setState(() => _errorMessage = '네트워크 오류가 발생했습니다.');
      print(error);
    } finally {
      setState(() => _isLoading = false); // 로딩스피너 해제
    }
  }

  /// 로그인 성공 시 사용자 정보 저장
  Future<void> saveUserData(Map<String, dynamic> userData) async {
    await storage.deleteAll(); // 기존 데이터 초기화
    userData = userData['body'];

    await storage.write(key: 'user_email', value: userData['email']);
    await storage.write(
      key: 'user_name',
      value: userData['nickname'].toString(),
    );
    await storage.write(key: 'access_token', value: userData['accessToken']);
    await storage.write(key: 'refresh_token', value: userData['refreshToken']);
  }

  @override
  Widget build(BuildContext context) {
    double bottomPadding = MediaQuery.of(context).padding.bottom;

    return SafeArea(
      child: Scaffold(
        backgroundColor: Theme.of(context).scaffoldBackgroundColor,
        body: SingleChildScrollView(
          physics: _keyboardIsVisible
              ? const AlwaysScrollableScrollPhysics()
              : const NeverScrollableScrollPhysics(),
          child: ConstrainedBox(
            constraints: BoxConstraints(
              minHeight: MediaQuery.of(context).size.height,
            ),
            child: Padding(
              padding: EdgeInsets.only(bottom: bottomPadding),
              child: Center(
                child: Stack(
                  children: [
                    Padding(
                      padding: EdgeInsets.symmetric(vertical: 25.h),
                      child: Column(
                        mainAxisAlignment: MainAxisAlignment.center,
                        crossAxisAlignment: CrossAxisAlignment.center,
                        children: [
                          Container(
                            alignment: Alignment.center,
                            child: Image.asset(
                              "assets/images/appLogo.png",
                              height: 85.h,
                            ),
                          ),
                          SizedBox(height: 30.h),
                          SizedBox(
                            width: 170.w, // 입력 필드의 최대 너비 설정
                            child: Column(
                              children: [
                                buildTextField(
                                  // 아이디 입력 필드
                                  controller: _emailController,
                                  hint: '아이디(이메일)',
                                  obscureText: false, // 가려지지 않음
                                  suffixIcon: null,
                                ),
                                SizedBox(height: 10.h),
                                buildTextField(
                                  // 비밀번호 입력 필드
                                  controller: _passwordController,
                                  hint: '비밀번호',
                                  obscureText:
                                      !_isPasswordVisible, // 비밀번호 보기 상태 기능 활성화
                                  suffixIcon: IconButton(
                                    // 눈 모양 아이콘 클릭하면 비밀번호 보이게 함
                                    icon: Icon(_isPasswordVisible
                                        ? Icons.visibility
                                        : Icons.visibility_off),
                                    onPressed: () {
                                      // 아이콘 클릭할 때마다 상태 변경
                                      setState(() => _isPasswordVisible =
                                          !_isPasswordVisible);
                                    },
                                  ),
                                ),
                                if (_errorMessage != null) _buildErrorMessage(),
                                SizedBox(height: 20.h),
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
                  ],
                ),
              ),
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
        padding: EdgeInsets.only(left: 3.w, top: 5.h),
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
      child: Center(
        child: Text(
          '로그인',
          style: TextStyle(fontSize: 7.5.sp, color: Colors.white),
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
          child: Text('비밀번호 찾기',
              style: TextStyle(fontSize: 5.5.sp, color: Colors.black54)),
        ),
        const Text('|'),
        TextButton(
          // '회원가입' 버튼
          onPressed: () => Navigator.push(
              context, MaterialPageRoute(builder: (_) => const SignUpScreen())),
          child: Text('회원가입',
              style: TextStyle(fontSize: 5.5.sp, color: Colors.black54)),
        ),
      ],
    );
  }
}
