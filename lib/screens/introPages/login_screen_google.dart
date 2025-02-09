import 'package:capstone_2025/screens/introPages/login_screen.dart';
import 'package:capstone_2025/screens/introPages/sign_up_screen.dart';
import 'package:flutter/material.dart';
import 'package:font_awesome_flutter/font_awesome_flutter.dart';
import 'package:google_sign_in/google_sign_in.dart';

class LoginScreenGoogle extends StatefulWidget {
  const LoginScreenGoogle({super.key});

  @override
  State<LoginScreenGoogle> createState() => _LoginScreenGoogleState();
}

class _LoginScreenGoogleState extends State<LoginScreenGoogle> {
  // GoogleSignIn 인스턴스 생성 (전역 변수)
  final GoogleSignIn _googleSignIn = GoogleSignIn(
    scopes: ['email'], // 이메일 정보 가져오기
  );

  // Google 로그인 실행 함수
  Future<void> _handleGoogleSignIn() async {
    try {
      print("Google 로그인 시도...");

      // ✅ 기존 로그인 계정 로그아웃
      await _googleSignIn.signOut();

      // ✅ 새로운 로그인 시도
      final GoogleSignInAccount? googleUser = await _googleSignIn.signIn();

      if (googleUser == null) {
        print("로그인 취소됨");
        return;
      }

      print("Google 로그인 성공!");
      print("이름: ${googleUser.displayName}");
      print("이메일: ${googleUser.email}");
      print("Google ID: ${googleUser.id}");
      print("프로필 사진: ${googleUser.photoUrl}");
    } catch (error) {
      print("Google 로그인 오류: $error");
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Theme.of(context).scaffoldBackgroundColor,
      body: Center(
        child: Padding(
          padding: EdgeInsets.symmetric(
            vertical: 70,
            horizontal: 20,
          ),
          child: Column(
            children: [
              SizedBox(
                width: 500,
                child: Text(
                  '🥁알려드럼🥁',
                  style: TextStyle(
                    fontSize: 45,
                    fontWeight: FontWeight.w800,
                  ),
                  textAlign: TextAlign.center,
                ),
              ),
              SizedBox(height: 40),
              SizedBox(
                width: 500,
                height: 60,
                child: ButtonForm(
                  btnName: "이메일로 로그인",
                  buttonColor: Color(0xFF424242),
                  clickedFunc: () {
                    Navigator.of(context).pushReplacement(
                      MaterialPageRoute(
                        builder: (_) => LoginScreen(),
                      ),
                    );
                  },
                ),
              ),
              SizedBox(height: 20),
              SizedBox(
                width: 500,
                height: 60,
                child: ButtonForm(
                  btnName: "구글 계정으로 로그인",
                  isTextBlack: true,
                  buttonColor: Color(0xFFE1E1E1),
                  needGoogle: true,
                  clickedFunc: _handleGoogleSignIn, // ✅ Google 로그인 함수 적용
                ),
              ),
              SizedBox(height: 15),
              Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Text(
                    "회원이 아니신가요?",
                    style: TextStyle(
                      fontWeight: FontWeight.w600,
                      color: Colors.black54,
                      fontSize: 17,
                    ),
                  ),
                  SizedBox(width: 15),
                  TextButton(
                    onPressed: () {
                      Navigator.of(context).pushReplacement(
                        MaterialPageRoute(
                          builder: (_) => SignUpScreen(),
                        ),
                      );
                    },
                    child: Text(
                      "회원가입",
                      style: TextStyle(
                        fontWeight: FontWeight.w600,
                        color: Colors.black54,
                        fontSize: 17,
                      ),
                    ),
                  )
                ],
              )
            ],
          ),
        ),
      ),
    );
  }
}

class ButtonForm extends StatelessWidget {
  ButtonForm({
    super.key,
    required this.btnName,
    this.buttonColor = const Color(0xFFD97D6C),
    this.isTextBlack = false,
    this.clickedFunc,
    this.needGoogle = false,
  });

  final String btnName;
  final Color buttonColor;
  final bool isTextBlack;
  final bool needGoogle;
  final VoidCallback? clickedFunc;

  @override
  Widget build(BuildContext context) {
    return ElevatedButton(
      style: ElevatedButton.styleFrom(
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(12.0),
        ),
        padding: EdgeInsets.symmetric(vertical: 16.5, horizontal: 10),
        backgroundColor: buttonColor,
      ),
      onPressed: clickedFunc,
      child: Center(
        child: Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            if (needGoogle)
              Icon(
                FontAwesomeIcons.google, // ✅ Google 아이콘 유지
                size: 20,
                color: Colors.black,
              ),
            SizedBox(width: 10),
            Text(
              btnName,
              style: TextStyle(
                fontSize: 15.0,
                color: isTextBlack ? Colors.black : Colors.white,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
