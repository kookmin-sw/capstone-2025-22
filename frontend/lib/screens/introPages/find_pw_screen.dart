import 'dart:async'; // 비동기 처리 및 타이머 기능을 위한 라이브러리
import 'dart:convert'; // JSON 데이터 변환을 위한 라이브러리
import 'package:capstone_2025/services/api_func.dart';
import 'package:flutter/material.dart'; // Flutter UI를 구성하는 라이브러리
import 'package:flutter_screenutil/flutter_screenutil.dart';
import 'package:http/http.dart' as http; // HTTP 요청을 보내기 위한 라이브러리
import 'package:flutter_secure_storage/flutter_secure_storage.dart'; // 보안 저장소를 사용하기 위한 라이브러리
import 'package:capstone_2025/screens/introPages/login_screen.dart'; // 로그인 화면 import
import 'package:capstone_2025/screens/introPages/set_new_pw_screen.dart'; // 비밀번호 재설정 화면 import
import 'package:capstone_2025/screens/introPages/widgets/intro_page_header.dart'; // 페이지 헤더 위젯 import

/// 비밀번호 변경 전, 사용자가 이메일을 통해 본인 인증을 수행하는 화면
class FindPwScreen extends StatefulWidget {
  // 뒤로가기 시에 돌아갈 화면
  final Widget? targetPage;
  const FindPwScreen({
    super.key,
    this.targetPage,
  });

  @override
  State<FindPwScreen> createState() => _FindPwScreenState();
}

class _FindPwScreenState extends State<FindPwScreen> {
  String emailToken = "";

  Timer? _timer; // 타이머 객체
  int _timeRemaining = 180; // 타이머 초기값(3분)
  bool _isTimerRunning = false; // 타이머가 실행 여부
  bool _isEmailSent = false; // 이메일 전송 여부
  bool _isCodeValid = false; // 인증번호 유효 여부

  String _emailMessage = ''; // 이메일 관련 메시지
  Color _emailMessageColor = Colors.red; // 이메일 메시지 색상
  String _codeMessage = ''; // 인증번호 관련 메시지
  Color _codeMessageColor = Colors.red; // 인증번호 메시지 색상

  final _emailController = TextEditingController(); // 이메일 입력 컨트롤러
  final _codeController = TextEditingController(); // 인증번호 입력 컨트롤러

  // 이메일 버튼 활성화 상태 변수
  bool _isEmailButtonEnabled = true;

  @override
  void dispose() {
    // 화면이 닫힐 때(위젯 종료 시) 호출: 타이머 종료 & 메모리 해제
    _timer?.cancel(); // 타이머 종료
    _emailController.dispose(); //컨트롤러 해제
    _codeController.dispose();
    super.dispose();
  }

  // 타이머 시작 함수
  void _startTimer() {
    setState(() {
      _isTimerRunning = true;
      _timeRemaining = 180; // 3분으로 초기화
    });

    _timer = Timer.periodic(const Duration(seconds: 1), (timer) {
      if (_timeRemaining > 0) {
        setState(() => _timeRemaining--); // 매 초마다 남은 시간 감소
      } else {
        // 시간이 0이 되면 _isTimerRunning을 false로 변경 (타이머 중지)
        _timer?.cancel(); // 타이머 종료
        setState(() {
          _isTimerRunning = false;
          _isEmailButtonEnabled = true; // 타이머 만료 시 이메일 버튼 활성화
        });
        _updateCodeMessage("인증번호 시간이 만료되었습니다. 이메일을 다시 전송해주세요.", Colors.red);
        _updateEmailMessage("", Colors.green); // 메시지 비우기
      }
    });
  }

  // 남은 시간을 (MM:SS) 포맷으로 변환하는 함수
  String _formatTime(int seconds) {
    final minutes = (seconds ~/ 60).toString().padLeft(2, '0');
    final secs = (seconds % 60).toString().padLeft(2, '0');
    return '$minutes:$secs'; // 'MM:SS' 형태로 변환
  }

  // 이메일 형식 유효성 검사 함수
  bool _validateEmail(String email) {
    return RegExp(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
        .hasMatch(email);
  }

  // 서버에서 이메일 가입 여부 확인 요청
  Future<bool> _isRegisteredEmail(String email) async {
    print("가입된 이메일인지 확인");
    // JSON 데이터 정의
    final Map<String, String> queryParam = {
      // API를 통해 전달할 param
      "email": email,
    };
    // 이메일 존재 여부 확인
    Map<String, dynamic> resData =
        await getHTTP("/verification/emails", queryParam);
    return resData['body'] == "invalid"; // 서버가 invalid이면 가입된 이메일
  }

  /// 이메일 인증번호 전송 버튼 눌렀을 때 실행되는 함수
  Future<void> _sendVerificationEmail() async {
    final email = _emailController.text.trim();
    _updateEmailMessage("", Colors.green); // 메시지 비우기
    _updateCodeMessage("", Colors.green); // 메시지 비우기

    // 예외처리1: 이메일 주소 형식이 잘못되었을 때
    if (!_validateEmail(email)) {
      _updateEmailMessage("이메일 주소 형식이 잘못됐습니다.", Colors.red);
      return;
    }

    // 예외처리2: 가입되지 않은 이메일일 때
    if (!await _isRegisteredEmail(email)) {
      _updateEmailMessage("가입되지 않은 이메일입니다.", Colors.red);
      return;
    }

    showLoadingDialog(context); // Show loading
    setState(() {
      _isEmailButtonEnabled = false;
    });

    final Map<String, String> queryParam = {
      // API를 통해 전달할 param
      "email": email,
    };

    // 인증 코드 전송
    Map<String, dynamic> resData =
        await getHTTP("/verification/auth-codes", queryParam);

    if (Navigator.canPop(context)) Navigator.pop(context); // Close loading

    if (resData["body"] == "valid") {
      print("이메일 전송 성공");
      _updateEmailMessage("이메일을 전송했습니다.", Colors.green);
      setState(() => _isEmailSent = true);
      _startTimer(); // 타이머 시작
    } else {
      _updateEmailMessage("이메일 전송에 실패했습니다.", Colors.red);
    }
    if (resData['errMessage'] != null) {
      // error 발생 시
      print(resData['errMessage']);
      return;
    }
  }

  /// 인증번호 확인 버튼을 눌렀을 때 실행되는 함수
  Future<void> _verifyCode() async {
    final email = _emailController.text.trim();
    final code = _codeController.text.trim();

    // 예외처리1: 이메일을 입력&전송하지 않았을 때
    if (!_isEmailSent) {
      _updateEmailMessage("이메일을 입력하고 전송 버튼을 눌러주세요", Colors.red);
      return;
    }

    // 예외처리2: 인증번호를 입력하지 않았을 때
    if (code.isEmpty) {
      _updateCodeMessage("인증번호를 입력해주세요.", Colors.red);
      return;
    }

    // JSON 데이터 정의
    final Map<String, String> queryParam = {
      // API를 통해 전달할 param
      "email": email,
      "authCode": code,
    };

    Map<String, dynamic> resData =
        await getHTTP("/verification/auth-codes/check", queryParam);

    if (resData["body"] != null) {
      // 이메일 토큰 받아서 저장
      emailToken = resData['body']['emailToken'];
      _updateCodeMessage("인증되었습니다.", Colors.green);
      setState(() {
        _isCodeValid = true;
        _timer?.cancel(); // 타이머 종료
      });
    } else {
      _updateCodeMessage("인증번호가 틀렸습니다.", Colors.red);
    }
  }

  // 이메일 메시지 업데이트 함수
  void _updateEmailMessage(String message, Color color) {
    setState(() {
      _emailMessage = message;
      _emailMessageColor = color;
    });
  }

  // 인증번호 메시지 업데이트 함수
  void _updateCodeMessage(String message, Color color) {
    setState(() {
      _codeMessage = message;
      _codeMessageColor = color;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      resizeToAvoidBottomInset: true,
      backgroundColor: Theme.of(context).scaffoldBackgroundColor, // 테마 공유해서 사용
      body: SingleChildScrollView(
        physics: const NeverScrollableScrollPhysics(),
        child: Center(
          child: Column(
            children: [
              introPageHeader(
                title: '비밀번호 재설정',
                targetPage: widget.targetPage ?? const LoginScreen(),
              ),
              SizedBox(height: 5.h),
              Text(
                "본인인증을 위해 가입하신 이메일 주소로 인증번호를 발송합니다.",
                style: TextStyle(
                    fontSize: 5.sp,
                    fontWeight: FontWeight.w500,
                    color: Colors.black54),
              ),
              SizedBox(height: 30.h),
              SizedBox(
                width: 170.w,
                child: Column(
                  children: [
                    _buildTextField(
                      controller: _emailController,
                      hint: '이메일',
                      buttonText: '전송',
                      onButtonPressed: _sendVerificationEmail,
                    ),
                    _buildMessageText(_emailMessage, _emailMessageColor),
                    SizedBox(height: 2.h),
                    _buildTextField(
                      controller: _codeController,
                      hint: '인증번호',
                      timerText:
                          _isTimerRunning ? _formatTime(_timeRemaining) : null,
                      buttonText: '확인',
                      onButtonPressed: _verifyCode,
                    ),
                    _buildMessageText(_codeMessage, _codeMessageColor),
                  ],
                ),
              ),
              SizedBox(height: 20.h),
              _buildNextButton(),
            ],
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
          flex: 9,
          child: Stack(
            children: [
              TextField(
                controller: controller,
                decoration: InputDecoration(
                  hintText: hint,
                  hintStyle: TextStyle(fontSize: 5.5.sp),
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
                    borderSide:
                        BorderSide(color: Color(0xFF424242), width: 2.0),
                  ),
                ),
              ),
              // 타이머 추가
              if (timerText != null)
                Positioned(
                  top: 20.h,
                  right: 7.w, // 타이머를 오른쪽으로 배치

                  child: Text(
                    timerText,
                    style: TextStyle(
                      fontSize: 6.sp,
                      color: Colors.red,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ),
            ],
          ),
        ),
        SizedBox(width: 5.w),
        Expanded(
          flex: 2,
          child: ElevatedButton(
            onPressed: (buttonText == "전송" && !_isEmailButtonEnabled)
                ? null
                : onButtonPressed,
            style: ElevatedButton.styleFrom(
              backgroundColor: const Color(0xFFCF8A7A),
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(12.0),
              ),
              padding: EdgeInsets.symmetric(vertical: 6.w),
            ),
            child: Text(
              buttonText,
              style: TextStyle(color: Colors.white),
            ),
          ),
        ),
      ],
    );
  }

  /// 메시지 표시 위젯
  Widget _buildMessageText(String message, Color color) {
    return Align(
      alignment: Alignment.centerLeft, // 왼쪽 정렬
      child: Padding(
        padding: EdgeInsets.only(left: 1.w, top: 1.h),
        child: Text(message,
            style: TextStyle(
                color: color, fontSize: 5.sp, fontWeight: FontWeight.w500)),
      ),
    );
  }

  // 다음 버튼
  Widget _buildNextButton() {
    return SizedBox(
      width: 135.w,
      child: ElevatedButton(
        style: ElevatedButton.styleFrom(
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(12.0),
          ),
          padding: EdgeInsets.symmetric(vertical: 20.h),
          backgroundColor: Colors.grey[800],
        ),
        onPressed: _isCodeValid
            ? () {
                Navigator.push(
                  context,
                  MaterialPageRoute(
                    builder: (context) => SetNewPwScreen(
                      emailToken: emailToken,
                    ), // '다음' 버튼 클릭하면 페이지 이동.
                  ),
                );
              }
            : null, // 인증 실패 시 버튼 비활성화
        child: Text(
          '다음',
          style: TextStyle(fontSize: 6.sp, color: Colors.white),
        ),
      ),
    );
  }
}

// 로딩 다이얼로그 함수
void showLoadingDialog(BuildContext context) {
  showDialog(
    context: context,
    barrierDismissible: false,
    builder: (context) {
      return const Dialog(
        backgroundColor: Colors.transparent,
        child: Center(
          child: CircularProgressIndicator(),
        ),
      );
    },
  );
}
