import 'dart:async';
import 'package:capstone_2025/screens/introPages/login_screen.dart';
import 'package:capstone_2025/screens/introPages/widgets/intro_page_header.dart';
import 'package:flutter/material.dart';
import 'dart:convert';
import 'package:http/http.dart' as http;

/// 사용자가 이메일을 통해 본인 인증을 수행하는 화면
class FindPwScreen extends StatefulWidget {
  const FindPwScreen({super.key});

  @override
  State<FindPwScreen> createState() => _FindPwScreenState();
}

class _FindPwScreenState extends State<FindPwScreen> {
  late Timer _timer;
  int _timeRemaining = 180; // 남은 시간 3분 (초 단위)
  bool _isTimerRunning = false; // 타이머가 실행 중인지 여부
  bool _isEmailSent = false; // 이메일이 전송되었는지 여부 (전송 버튼 활성화 여부)
  bool _isCodeValid = false; // 사용자가 입력한 인증번호가 유효한지 여부
  final TextEditingController _emailController =
      TextEditingController(); // 이메일 입력 필드 제어하는 컨트롤러
  final TextEditingController _codeController =
      TextEditingController(); // 인증번호 입력 필드 제어하는 컨트롤러

  @override
  void dispose() {
    // 위젯 종료 시 실행
    // 화면이 닫힐 때 호출: 타이머 종료 & 메모리 해제
    _timer.cancel(); // 타이머 종료
    _emailController.dispose(); // 이메일 입력 컨트롤러 해제
    _codeController.dispose(); // 인증번호 입력 컨트롤러 해제
    super.dispose();
  }

  // 이메일 유효성 검사
  bool _validateEmail(String email) {
    final emailRegex = RegExp(
        r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'); // 이메일 형식 확인
    return emailRegex.hasMatch(email);
  }

  // 타이머 시작 함수: 이메일 전송 후 자동으로 3분 타이머 시작
  void _startTimer() {
    setState(() {
      _isTimerRunning = true;
      _timeRemaining = 180; // 3분으로 초기화
    });

    _timer = Timer.periodic(const Duration(seconds: 1), (timer) {
      if (_timeRemaining > 0) {
        // 1초마다 _timeRemaining 값을 1씩 감소
        setState(() {
          _timeRemaining--;
        });
      } else {
        // 시간이 0이 되면 _isTimerRunning을 false로 변경 (타이머 중지)
        _timer.cancel();
        setState(() {
          _isTimerRunning = false;
        });
      }
    });
  }

  // 타이머 형식 변환 함수 (초 → MM:SS)
  String _formatTime(int seconds) {
    final minutes = (seconds ~/ 60).toString().padLeft(2, '0');
    final secs = (seconds % 60).toString().padLeft(2, '0');
    return '$minutes:$secs';
  }

  // 이메일 인증번호 전송 요청
  Future<void> _sendEmailVerification() async {
    final email = _emailController.text.trim();

    if (!_validateEmail(email)) return; // 이메일 유효성 검사

    final uri =
        Uri.parse('http://10.0.2.2:28080/verification/auth-codes?email=$email');
    final response = await http.get(uri);

    final data = jsonDecode(response.body);

    if (data['body'] == "SUCCESS") {
      print("이메일 전송 성공");
      setState(() {
        _isEmailSent = true;
      });
      _startTimer(); // 이메일 전송되면 타이머 시작
    } else {
      print("이메일 전송 실패: ${response.body}");
    }
  }

  // 입력한 인증번호 확인 요청
  Future<void> _verifyCode() async {
    final email = _emailController.text.trim();
    final code = _codeController.text.trim();

    if (email.isEmpty || code.isEmpty) {
      print("이메일 또는 인증번호가 비어 있음.");
      return;
    }

    final uri = Uri.parse(
        'http://10.0.2.2:28080/verification/auth-codes/check?email=$email&authCode=$code');
    final response = await http.get(uri);

    final data = jsonDecode(response.body);

    print(response.body);

    if (data['body'] == "SUCCESS") {
      setState(() {
        _isCodeValid = true;
      });
    } else {
      setState(() {
        _isCodeValid = false; // 인증 실패
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
              mainAxisAlignment: MainAxisAlignment.center,
              crossAxisAlignment: CrossAxisAlignment.center,
              children: [
                introPageHeader(
                  title: '비밀번호 재설정',
                  targetPage: LoginScreen(),
                ),
                Text("본인인증을 위해 가입하신 이메일 주소로 인증번호를 발송합니다."),
                SizedBox(height: 30),
                SizedBox(
                  width: 400,
                  child: Column(
                    children: [
                      _buildTextFieldWithButton(
                        controller: _emailController,
                        hint: '이메일',
                        buttonText: '전송',
                        isButtonEnabled: _validateEmail(_emailController.text),
                        onButtonPressed: _sendEmailVerification,
                      ),
                      if (!_validateEmail(_emailController.text))
                        Align(
                          alignment: Alignment.centerLeft, // 왼쪽 정렬
                          child: Padding(
                            padding: const EdgeInsets.only(left: 8.0),
                            child: Text(
                              "이메일 주소 형식이 잘못됐습니다.",
                              style: TextStyle(color: Colors.red),
                            ),
                          ),
                        ),
                      SizedBox(height: 10),
                      _buildTextFieldWithTimerAndButton(
                        controller: _codeController,
                        hint: '인증번호',
                        timerText: _formatTime(_timeRemaining),
                        buttonText: '확인',
                        isButtonEnabled: _isEmailSent,
                        onButtonPressed: _verifyCode,
                      ),
                      if (!_isCodeValid)
                        Align(
                          alignment: Alignment.centerLeft, // 왼쪽 정렬
                          child: Padding(
                            padding: const EdgeInsets.only(left: 8.0),
                            child: Text(
                              "인증번호가 틀렸습니다.",
                              style: TextStyle(color: Colors.red),
                            ),
                          ),
                        ),
                    ],
                  ),
                ),
                SizedBox(height: 10),
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
                    onPressed: _isCodeValid // 인증번호가 유효할 때만 다음 버튼 활성화
                        ? () {
                            Navigator.push(
                              context,
                              MaterialPageRoute(
                                builder: (context) =>
                                    const FindPwScreen(), // '다음' 버튼 클릭하면 페이지 이동. 수정하기!
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

  // 입력 칸 오른쪽에 버튼 추가 - 이메일 입력 필드
  Widget _buildTextFieldWithButton({
    required TextEditingController controller,
    required String hint,
    required String buttonText,
    required VoidCallback onButtonPressed,
    required bool isButtonEnabled,
  }) {
    return Row(
      children: [
        Expanded(
          child: TextField(
            controller: controller,
            onChanged: (value) {
              // 입력 값이 변경될 때마다 버튼 상태 변경
              setState(() {});
            },
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
            ),
          ),
        ),
        SizedBox(width: 10),
        ElevatedButton(
          onPressed:
              isButtonEnabled ? onButtonPressed : null, // 이메일이 유효할 때만 버튼 활성화
          style: ElevatedButton.styleFrom(
            backgroundColor: isButtonEnabled ? Color(0xFFCF8A7A) : Colors.grey,
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

  // 입력 칸 내부에 타이머와 오른쪽에 버튼 추가 - 인증번호 입력 필드
  Widget _buildTextFieldWithTimerAndButton({
    required TextEditingController controller,
    required String hint,
    required String timerText,
    required String buttonText,
    required VoidCallback onButtonPressed,
    required bool isButtonEnabled,
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
              suffix: Text(
                // 입력 필드 내부에 타이머 표시
                timerText,
                style: TextStyle(
                    fontSize: 15,
                    color: Color(0xFFCF8A7A),
                    fontWeight: FontWeight.bold),
              ),
            ),
          ),
        ),
        SizedBox(width: 10),
        ElevatedButton(
          onPressed: isButtonEnabled ? onButtonPressed : null,
          style: ElevatedButton.styleFrom(
            backgroundColor: isButtonEnabled ? Color(0xFFCF8A7A) : Colors.grey,
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
