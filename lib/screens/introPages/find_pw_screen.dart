import 'dart:async';
import 'package:capstone_2025/screens/introPages/widgets/intro_page_header.dart';
import 'package:flutter/material.dart';

class FindPwScreen extends StatefulWidget {
  const FindPwScreen({super.key});

  @override
  State<FindPwScreen> createState() => _FindPwScreenState();
}

class _FindPwScreenState extends State<FindPwScreen> {
  late Timer _timer;
  int _timeRemaining = 180; // 3분 (초 단위)

  @override
  void initState() {
    super.initState();
    _startTimer(); // 타이머 시작
  }

  @override
  void dispose() {
    _timer.cancel(); // 타이머 종료
    super.dispose();
  }

  // 타이머 시작 함수
  void _startTimer() {
    _timer = Timer.periodic(Duration(seconds: 1), (timer) {
      if (_timeRemaining > 0) {
        setState(() {
          _timeRemaining--;
        });
      } else {
        _timer.cancel(); // 타이머 종료
      }
    });
  }

  // 타이머 형식 변환 함수 (초 → MM:SS)
  String _formatTime(int seconds) {
    final minutes = (seconds ~/ 60).toString().padLeft(2, '0');
    final secs = (seconds % 60).toString().padLeft(2, '0');
    return '$minutes:$secs';
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
                  title: '본인 확인',
                ),
                Text("본인인증을 위해 가입하신 이메일 주소로 인증번호를 발송합니다."),
                SizedBox(height: 30),
                SizedBox(
                  width: 400,
                  child: Column(
                    children: [
                      _buildTextFieldWithButton(
                        hint: '이메일',
                        buttonText: '전송',
                        onButtonPressed: () {
                          // 이메일 전송 로직
                        },
                      ),
                      SizedBox(height: 10),
                      _buildTextFieldWithTimerAndButton(
                        hint: '인증번호',
                        timerText: _formatTime(_timeRemaining),
                        buttonText: '확인',
                        onButtonPressed: () {
                          // 인증번호 확인 로직
                        },
                      ),
                    ],
                  ),
                ),
                SizedBox(height: 10),
                SizedBox(
                  width: 400,
                  child: ElevatedButton(
                    style: ElevatedButton.styleFrom(
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(12.0),
                      ),
                      padding: EdgeInsets.symmetric(vertical: 16.0),
                      backgroundColor: Color(0xFF424242),
                    ),
                    onPressed: () {
                      Navigator.push(
                        context,
                        MaterialPageRoute(
                          builder: (context) =>
                              const FindPwScreen(), // '다음' 버튼 클릭하면 페이지 이동. 수정하기!
                        ),
                      );
                    },
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

  // 입력 칸 오른쪽에 버튼 추가
  Widget _buildTextFieldWithButton({
    required String hint,
    required String buttonText,
    required VoidCallback onButtonPressed,
  }) {
    return Row(
      children: [
        Expanded(
          child: TextField(
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
          style: ElevatedButton.styleFrom(
            backgroundColor: Color(0xFFCF8A7A),
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(12.0),
            ),
            padding: EdgeInsets.symmetric(horizontal: 20.0, vertical: 16.0),
          ),
          onPressed: onButtonPressed,
          child: Text(
            buttonText,
            style: TextStyle(color: Colors.white),
          ),
        ),
      ],
    );
  }

  // 입력 칸 내부에 타이머와 오른쪽에 버튼 추가
  Widget _buildTextFieldWithTimerAndButton({
    required String hint,
    required String timerText,
    required String buttonText,
    required VoidCallback onButtonPressed,
  }) {
    return Row(
      children: [
        Expanded(
          child: TextField(
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
          style: ElevatedButton.styleFrom(
            backgroundColor: Color(0xFFCF8A7A),
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(12.0),
            ),
            padding: EdgeInsets.symmetric(horizontal: 20.0, vertical: 16.0),
          ),
          onPressed: onButtonPressed,
          child: Text(
            buttonText,
            style: TextStyle(color: Colors.white),
          ),
        ),
      ],
    );
  }
}
