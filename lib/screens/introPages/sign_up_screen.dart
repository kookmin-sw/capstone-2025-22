import 'package:capstone_2025/screens/introPages/widgets/build_text_field.dart';
import 'package:capstone_2025/screens/introPages/widgets/intro_page_header.dart';
import 'package:flutter/material.dart';

class SignUpScreen extends StatefulWidget {
  const SignUpScreen({super.key});

  @override
  State<SignUpScreen> createState() => _SignUpScreenState();
}

void goToMain() {
  return;
}

void clickButton() {
  return;
}

class _SignUpScreenState extends State<SignUpScreen> {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Theme.of(context).scaffoldBackgroundColor,
      body: Padding(
        padding: const EdgeInsets.symmetric(
          horizontal: 30,
          vertical: 10,
        ),
        child: SingleChildScrollView(
          scrollDirection: Axis.vertical,
          child: Column(
            children: [
              // 뒤로가기 버튼, 제목
              introPageHeader(
                title: '회원가입',
              ),
              // 나머지 입력 폼
              InputForm(
                name: '아이디',
                hintTxt: '본인 인증을 위한 이메일 주소를 입력해주세요.',
                needBtn: true,
                buttonName: '전송',
              ),
              Text(
                '이메일 주소 형식이 잘못됐습니다.' '이미 가입된 이메일 주소입니다.',
                style: TextStyle(
                    fontSize: 12,
                    color: Colors.red,
                    fontWeight: FontWeight.w600),
              ), // 조건 추가하기
              SizedBox(
                height: 15,
              ),
              InputForm(
                name: '인증번호',
                hintTxt: '인증번호 6자리를 입력해주세요.',
                needBtn: true,
                buttonName: '확인',
              ),
              SizedBox(
                height: 15,
              ),
              InputForm(
                name: '닉네임',
                hintTxt: '2 ~ 8자',
                needBtn: true,
                buttonName: '중복확인',
              ),
              SizedBox(
                height: 15,
              ),
              InputForm(
                  name: '비밀번호',
                  hintTxt: '8~20자의 영문, 숫자, 특수문자 조합',
                  needBtn: false),
              SizedBox(
                height: 15,
              ),
              InputForm(
                name: '비밀번호 확인',
                hintTxt: '비밀번호를 한 번 더 입력해주세요.',
                needBtn: false,
              ),
              SizedBox(
                height: 35,
              ),
              SizedBox(
                width: 200,
                child: ButtonForm(
                  buttonName: '제출',
                  buttonColor: Color(0xFF424242),
                ),
              )
            ],
          ),
        ),
      ),
    );
  }
}

class InputForm extends StatelessWidget {
  final String name, hintTxt, buttonName;
  final bool needBtn;

  InputForm({
    required this.name,
    required this.hintTxt,
    required this.needBtn,
    this.buttonName = "",
  });

  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.center,
      crossAxisAlignment: CrossAxisAlignment.center,
      children: [
        Expanded(
          flex: 1,
          child: SizedBox(),
        ),
        Expanded(
          flex: 6,
          child: Text(
            name,
            style: TextStyle(
                fontSize: 18, fontWeight: FontWeight.w600, color: Colors.black),
            textAlign: TextAlign.end,
          ),
        ),
        Expanded(
          flex: 1,
          child: SizedBox(
            width: 5,
          ),
        ),
        Expanded(
          flex: 9,
          child: buildTextField(
            hint: hintTxt,
            obscureText: false,
            suffixIcon: null,
          ),
        ),
        Expanded(
          flex: 1,
          child: SizedBox(
            width: 20,
          ),
        ),
        if (needBtn)
          Expanded(
            flex: 2,
            child: ButtonForm(buttonName: buttonName),
          ),
        Expanded(
          flex: needBtn ? 5 : 7,
          child: SizedBox(),
        )
      ],
    );
  }
}

class ButtonForm extends StatelessWidget {
  const ButtonForm(
      {super.key,
      required this.buttonName,
      this.buttonColor = const Color(0xFFD97D6C),
      this.isTextBlack = false});

  final String buttonName;
  final Color buttonColor;
  final bool isTextBlack;

  @override
  Widget build(BuildContext context) {
    return ElevatedButton(
      style: ElevatedButton.styleFrom(
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(12.0), // 버튼 테두리 둥글게
        ),
        padding: EdgeInsets.symmetric(vertical: 15, horizontal: 25), // 버튼 크기 설정
        backgroundColor: buttonColor, // 버튼 배경색
      ),
      onPressed: () {
        // 로그인 버튼 클릭
      },
      child: Center(
        child: Text(
          buttonName,
          style: TextStyle(
            fontSize: 15.0,
            color: isTextBlack ? Colors.black : Colors.white,
          ),
        ),
      ),
    );
  }
}
