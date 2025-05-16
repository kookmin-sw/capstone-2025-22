import 'dart:convert';
import 'package:capstone_2025/screens/introPages/login_screen.dart';
import 'package:capstone_2025/screens/introPages/sign_up_screen.dart';
import 'package:capstone_2025/screens/mainPages/navigation_screens.dart';
import 'package:capstone_2025/services/api_func.dart';
import 'package:capstone_2025/services/server_addr.dart';
import '/services/storage_service.dart';
import 'package:flutter/material.dart';
import 'package:flutter_screenutil/flutter_screenutil.dart';
import 'package:font_awesome_flutter/font_awesome_flutter.dart';
import 'package:google_sign_in/google_sign_in.dart';
import 'package:http/http.dart' as http;

class LoginScreenGoogle extends StatefulWidget {
  // Google 로그인 포함되어 있는 로그인 페이지

  const LoginScreenGoogle({super.key});

  @override
  State<LoginScreenGoogle> createState() => _LoginScreenGoogleState();
}

class _LoginScreenGoogleState extends State<LoginScreenGoogle> {
  // GoogleSignIn 인스턴스 생성 (전역 변수)
  final GoogleSignIn _googleSignIn = GoogleSignIn(
    scopes: ['email', 'profile', 'openid'], // Google 로그인 시 필요한 scope
    serverClientId: // web client id
        "181628362307-2o025ta4bgqqqdtki2f5r6bkmerh9722.apps.googleusercontent.com",
    forceCodeForRefreshToken: true, // refresh token 받기 위해 필요
  );

  // Google 로그인 실행 함수
  Future<void> _handleGoogleSignIn() async {
    try {
      // print("Google 로그인 시도..."); // 디버깅용

      // 기존 로그인 계정 로그아웃
      // await _googleSignIn.signOut();

      // 로그인 시도
      final GoogleSignInAccount? googleUser = await _googleSignIn.signIn();

      if (googleUser == null) {
        // 사용자가 로그인 창 나갔을 때
        print("로그인 취소됨");
        return;
      }

      print("Google 로그인 성공!");
      print("이메일: ${googleUser.email}");

      //auth code 받아오는 코드
      final String? authCode = googleUser.serverAuthCode;

      if (authCode == null) {
        // auth code 받아오기 실패했을 때
        print("Auth code를 가져올 수 없습니다.");
        return;
      }

      print("Auth Code: $authCode");

      final Map<String, dynamic> requestBody = {
        // HTTP 함수를 통해 보낼 request body
        "googleAuthCode": authCode,
      };

      final userInfo = await postHTTP("/auth/signin/google", requestBody);
      if (userInfo['errMessage'] == null) {
        // 로그인 성공
        print("UserInfo: $userInfo");
        saveUserInfo(userInfo); // 정보 저장
        Navigator.of(context).pushReplacement(
          // 메인 페이지로 이동
          MaterialPageRoute(
            builder: (_) => NavigationScreens(),
          ),
        );
      } else {
        // 로그인 실패
        print("Google 로그인 실패: ${userInfo['errMessage']}");
      }
    } catch (error) {
      // 로그인 중 오류 발생
      print("Google 로그인 오류: $error");
    }
  }

  // Response 받은 정보들 저장하는 함수
  Future<void> saveUserInfo(Map<String, dynamic> userInfo) async {
    await storage.write(key: "user_email", value: userInfo["email"]);
    await storage.write(
      key: 'user_name',
      value: utf8.decode(userInfo["name"].toString().codeUnits),
    );
    await storage.write(key: "access_token", value: userInfo["access_token"]);
    await storage.write(key: "refresh_token", value: userInfo["refresh_token"]);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Theme.of(context).scaffoldBackgroundColor, // 배경색
      body: SingleChildScrollView(
        physics: const NeverScrollableScrollPhysics(),
        child: Center(
          child: Padding(
            padding: EdgeInsets.symmetric(
              vertical: 80.h,
              horizontal: 50.w,
            ),
            child: Column(
              children: [
                SizedBox(
                  child: Container(
                    alignment: Alignment.center,
                    child: Image.asset(
                      "assets/images/appLogo.png",
                      height: 95.h,
                    ),
                  ),
                ),
                SizedBox(height: 40.h),
                SizedBox(
                  // 이메일 로그인 버튼
                  width: 190.w,
                  height: 70.h,
                  child: ButtonForm(
                    btnName: "이메일로 로그인",
                    buttonColor: Color(0xFF424242),
                    clickedFunc: () {
                      Navigator.of(context).push(
                        MaterialPageRoute(
                          // 이메일 로그인 페이지로 이동
                          builder: (_) => LoginScreen(),
                        ),
                      );
                    },
                  ),
                ),
                SizedBox(height: 20.h), // 간격
                SizedBox(
                  // 구글 로그인 버튼
                  width: 190.w,
                  height: 70.h,
                  child: ButtonForm(
                    btnName: "구글 계정으로 로그인",
                    isTextBlack: true,
                    buttonColor: Colors.white,
                    needGoogle: true,
                    clickedFunc: _handleGoogleSignIn, // Google 로그인 함수 적용
                  ),
                ),
                SizedBox(height: 20.h),
                Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Text(
                      "회원이 아니신가요?",
                      style: TextStyle(
                        fontWeight: FontWeight.w600,
                        color: Colors.black54,
                        fontSize: 6.sp,
                      ),
                    ),
                    SizedBox(width: 3.w),
                    TextButton(
                      // 회원가입 버튼
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
                          fontWeight: FontWeight.w500,
                          color: Colors.black54,
                          fontSize: 6.5.sp,
                        ),
                      ),
                    )
                  ],
                )
              ],
            ),
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
    this.needGoogle = false, // 구글 이모지 필요 여부
  });

  final String btnName;
  final Color buttonColor;
  final bool isTextBlack;
  final bool needGoogle;
  final VoidCallback? clickedFunc;

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: BoxDecoration(
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.1),
            offset: Offset(0, 4),
            blurRadius: 2.0,
          ),
        ],
        borderRadius: BorderRadius.circular(12.0),
      ),
      child: ElevatedButton(
        style: ElevatedButton.styleFrom(
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(12.0),
          ),
          padding: EdgeInsets.symmetric(vertical: 16.5.h, horizontal: 10.w),
          backgroundColor: buttonColor,
          shadowColor: Colors.transparent, // Prevent double shadows
        ),
        onPressed: clickedFunc,
        child: Center(
          child: Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              if (needGoogle)
                Image.asset("assets/images/googleLogo.png", height: 32.h),
              SizedBox(width: 5.w),
              Text(
                btnName,
                style: TextStyle(
                  fontSize: 6.sp,
                  color: isTextBlack ? Colors.black54 : Colors.white,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
