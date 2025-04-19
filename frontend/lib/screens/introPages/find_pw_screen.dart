import 'dart:async';
import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:capstone_2025/screens/introPages/login_screen.dart';
import 'package:capstone_2025/screens/introPages/set_new_pw_screen.dart';
import 'package:capstone_2025/screens/introPages/widgets/intro_page_header.dart';

/// 사용자가 이메일을 통해 본인 인증을 수행하는 화면
class FindPwScreen extends StatefulWidget {
  const FindPwScreen({super.key});

  @override
  State<FindPwScreen> createState() => _FindPwScreenState();
}

class _FindPwScreenState extends State<FindPwScreen> {
  Timer? _timer;
  int _timeRemaining = 180; // 타이머 초기값(3분)
  bool _isTimerRunning = false; // 타이머가 실행 중인지 여부
  bool _isEmailSent = false; // 이메일이 전송되었는지 여부
  bool _isCodeValid = false; // 인증번호가 유효한지 여부

  String _emailMessage = '';
  Color _emailMessageColor = Colors.red;
  String _codeMessage = '';
  Color _codeMessageColor = Colors.red;

  final TextEditingController _emailController = TextEditingController();
  final TextEditingController _codeController = TextEditingController();

  @override
  void dispose() {
    // 화면이 닫힐 때(위젯 종료 시) 호출: 타이머 종료 & 메모리 해제
    _timer?.cancel(); // 타이머 종료
    _emailController.dispose(); //컨트롤러 해제
    _codeController.dispose();
    super.dispose();
  }

  // 이메일 형식 유효성 검사
  bool _validateEmail(String email) {
    final emailRegex =
        RegExp(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$');
    return emailRegex.hasMatch(email);
  }

  // 서버에서 이메일 가입 여부 확인
  Future<bool> _isRegisteredEmail(String email) async {
    final uri = Uri.parse('http://10.0.2.2:28080/verification/emails=$email');
    final response = await http.get(uri);
    return response.statusCode == 200; // 200이면 등록된 이메일
  }

  // 타이머 시작
  void _startTimer() {
    setState(() {
      _isTimerRunning = true;
      _timeRemaining = 180; // 3분으로 초기화
    });

    _timer = Timer.periodic(const Duration(seconds: 1), (timer) {
      if (_timeRemaining > 0) {
        setState(() => _timeRemaining--);
      } else {
        // 시간이 0이 되면 _isTimerRunning을 false로 변경 (타이머 중지)
        _timer?.cancel();
        setState(() => _isTimerRunning = false);
      }
    });
  }

  // 남은 시간 포맷 (MM:SS)
  String _formatTime(int seconds) {
    final minutes = (seconds ~/ 60).toString().padLeft(2, '0');
    final secs = (seconds % 60).toString().padLeft(2, '0');
    return '$minutes:$secs';
  }

  /// 이메일 인증번호 전송 버튼 눌렀을 때 실행되는 함수
  Future<void> _sendVerificationEmail() async {
    final email = _emailController.text.trim();

    // 예외처리1: 이메일 주소 형식이 잘못되었을 때
    if (!_validateEmail(email)) {
      setState(() {
        _emailMessage = "이메일 주소 형식이 잘못됐습니다.";
        _emailMessageColor = Colors.red;
      });
      return;
    }

    // 예외처리2: 가입되지 않은 이메일일 때
    if (!await _isRegisteredEmail(email)) {
      setState(() {
        _emailMessage = "가입되지 않은 이메일입니다.";
        _emailMessageColor = Colors.red;
      });
      return;
    }

    final uri =
        Uri.parse('http://10.0.2.2:28080/verification/auth-codes?email=$email');
    final response = await http.get(uri);
    final data = jsonDecode(response.body);

    if (response.statusCode == 200 && data['body'] == "SUCCESS") {
      print("이메일 전송 성공");
      setState(() {
        _isEmailSent = true;
        _emailMessage = '이메일을 전송했습니다.';
        _emailMessageColor = Colors.green;
      });
      _startTimer(); // 타이머 시작
    } else {
      setState(() {
        _emailMessage = '이메일 전송에 실패했습니다.';
        _emailMessageColor = Colors.red;
      });
    }
  }

  /// 인증번호 확인 버튼을 눌렀을 때 실행되는 함수
  Future<void> _verifyCode() async {
    final email = _emailController.text.trim();
    final code = _codeController.text.trim();

    // 예외처리1: 이메일을 입력&전송하지 않았을 때
    if (!_isEmailSent) {
      setState(() {
        _emailMessage = "이메일을 입력하고 전송 버튼을 눌러주세요";
        _emailMessageColor = Colors.red;
      });
      return;
    }

    // 예외처리2: 인증번호를 입력하지 않았을 때
    if (code.isEmpty) {
      setState(() {
        _codeMessage = "인증번호를 입력해주세요.";
        _codeMessageColor = Colors.red;
      });
      return;
    }

    final uri = Uri.parse(
        'http://10.0.2.2:28080/verification/auth-codes/check?email=$email&authCode=$code');
    final response = await http.get(uri);
    final data = jsonDecode(response.body);

    if (response.statusCode == 200 && data['body'] == 'SUCCESS') {
      setState(() {
        _isCodeValid = true;
        _codeMessage = '인증되었습니다.';
        _codeMessageColor = Colors.green;
      });
      _timer?.cancel(); // 타이머 종료
    } else {
      setState(() {
        _isCodeValid = false; // 인증 실패
        _codeMessage = "인증번호가 틀렸습니다.";
        _codeMessageColor = Colors.red;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Theme.of(context).scaffoldBackgroundColor, // 테마 공유해서 사용
      body: Center(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 20.0),
          child: SingleChildScrollView(
            child: Column(
              children: [
                introPageHeader(title: '본인 확인', targetPage: LoginScreen()),
                const Text("본인인증을 위해 가입하신 이메일 주소로 인증번호를 발송합니다."),
                const SizedBox(height: 30),
                SizedBox(
                  width: 450,
                  child: Column(
                    children: [
                      _buildTextField(
                        controller: _emailController,
                        hint: '이메일',
                        buttonText: '전송',
                        onButtonPressed: _sendVerificationEmail,
                      ),
                      Align(
                        alignment: Alignment.centerLeft, // 왼쪽 정렬
                        child: Padding(
                          padding: const EdgeInsets.only(left: 8.0),
                          child: Text(
                            _emailMessage,
                            style: TextStyle(color: _emailMessageColor),
                          ),
                        ),
                      ),
                      SizedBox(height: 5),
                      _buildTextField(
                        controller: _codeController,
                        hint: '인증번호',
                        timerText: _formatTime(_timeRemaining),
                        buttonText: '확인',
                        onButtonPressed: _verifyCode,
                      ),
                      Align(
                        alignment: Alignment.centerLeft, // 왼쪽 정렬
                        child: Padding(
                          padding: const EdgeInsets.only(left: 8.0),
                          child: Text(
                            _codeMessage,
                            style: TextStyle(color: _codeMessageColor),
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
                if (!_isTimerRunning && _isEmailSent) // 타이머가 종료되고 이메일이 전송되었을 경우
                  Text(
                    "인증번호 시간이 만료되었습니다. 이메일을 다시 전송해주세요.",
                    style: TextStyle(color: Colors.red),
                  ),
                const SizedBox(height: 20),
                SizedBox(
                  width: 300,
                  child: ElevatedButton(
                    style: ElevatedButton.styleFrom(
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(12.0),
                      ),
                      padding: EdgeInsets.symmetric(vertical: 16.0),
                      backgroundColor: Color(0xFF424242),
                    ),
                    onPressed: _isCodeValid
                        ? () {
                            Navigator.push(
                              context,
                              MaterialPageRoute(
                                builder: (context) =>
                                    const SetNewPwScreen(), // '다음' 버튼 클릭하면 페이지 이동.
                              ),
                            );
                          }
                        : null, // 인증 실패 시 버튼 비활성화
                    child: Center(
                      child: Text(
                        '다음',
                        style: TextStyle(
                          fontSize: 15.0,
                          color: Colors.white,
                        ),
                      ),
                    ),
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  // 입력 필드
  Widget _buildTextField({
    required TextEditingController controller,
    required String hint,
    required String buttonText,
    required VoidCallback onButtonPressed,
    String? timerText,
  }) {
    return Row(
      children: [
        Expanded(
          child: TextField(
            controller: controller,
            decoration: InputDecoration(
              hintText: hint,
              hintStyle: TextStyle(fontSize: 15),
              filled: true,
              fillColor: Colors.white,
              border: OutlineInputBorder(
                borderRadius: BorderRadius.circular(12.0),
              ),
              enabledBorder: OutlineInputBorder(
                borderRadius: BorderRadius.circular(12.0),
                borderSide: BorderSide(color: Colors.grey.shade400),
              ),
              focusedBorder: OutlineInputBorder(
                borderRadius: BorderRadius.circular(12.0),
                borderSide: BorderSide(color: Color(0xFF424242), width: 2.0),
              ),
              suffix: timerText != null
                  ? Text(
                      // 입력 필드 내부에 타이머 표시
                      timerText,
                      style: TextStyle(
                          fontSize: 15,
                          color: Color(0xFFCF8A7A),
                          fontWeight: FontWeight.bold),
                    )
                  : null,
            ),
          ),
        ),
        SizedBox(width: 10),
        ElevatedButton(
          onPressed: onButtonPressed,
          style: ElevatedButton.styleFrom(
            backgroundColor: Color(0xFFCF8A7A),
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(12.0),
            ),
            padding: EdgeInsets.symmetric(horizontal: 20.0, vertical: 16.0),
          ),
          child: Text(
            buttonText,
            style: TextStyle(color: Colors.white),
          ),
        ),
      ],
    );
  }
}
