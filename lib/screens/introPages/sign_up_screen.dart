// 입력창에 입력 안됨
// 버튼 함수 제대로 연결 안됨
// 버튼 함수 상세 정의 안함
// 제출 버튼 연결 안함
// 뒤로 가기 버튼 동작 안함
// API 여기서 사용하는 것 같은데 알아보기

import 'package:capstone_2025/screens/introPages/widgets/build_text_field.dart';
import 'package:capstone_2025/screens/introPages/widgets/intro_page_header.dart';
import 'package:flutter/material.dart';

class SignUpScreen extends StatefulWidget {
  const SignUpScreen({super.key});

  @override
  State<SignUpScreen> createState() => _SignUpScreenState();
}

class _SignUpScreenState extends State<SignUpScreen> {
  bool isCorrectEmail = true;
  bool isAlreadyUseEmail = true;
  bool isCorrectNum = true;
  bool isAlreadyUseName = true;
  bool isCorrectFormat = true;
  bool isCorrectPw = true;

  void submitBtn() {
    return;
  }

  void idAuth() {
    setState(() {
      isAlreadyUseEmail = !isAlreadyUseEmail;
    });
    print(isAlreadyUseEmail);
  }

  void numAuth() {
    setState(() {
      isCorrectNum = !isCorrectNum;
    });
    print(isCorrectNum);
  }

  void nameAuth() {
    setState(() {
      isAlreadyUseName = !isAlreadyUseName;
    });
    print(isAlreadyUseName);
  }

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
                btnName: '전송',
                clickedFunc: idAuth,
              ),
              if (!isCorrectEmail)
                Text(
                  '이메일 주소 형식이 잘못됐습니다.',
                  style: TextStyle(
                    fontSize: 12,
                    color: Colors.red,
                    fontWeight: FontWeight.w600,
                  ),
                ),
              if (!isAlreadyUseEmail)
                Text(
                  '이미 가입된 이메일 주소입니다.',
                  style: TextStyle(
                      fontSize: 12,
                      color: Colors.red,
                      fontWeight: FontWeight.w600),
                ),
              SizedBox(
                height: 15,
              ),
              InputForm(
                name: '인증번호',
                hintTxt: '인증번호 6자리를 입력해주세요.',
                needBtn: true,
                btnName: '확인',
                clickedFunc: numAuth,
              ),
              if (!isCorrectNum)
                Text(
                  '인증번호가 틀렸습니다.',
                  style: TextStyle(
                      fontSize: 12,
                      color: Colors.red,
                      fontWeight: FontWeight.w600),
                ),
              SizedBox(
                height: 15,
              ),
              InputForm(
                name: '닉네임',
                hintTxt: '2 ~ 8자',
                needBtn: true,
                btnName: '중복확인',
                clickedFunc: nameAuth,
              ),
              if (!isAlreadyUseName)
                Text(
                  '이미 가입된 닉네임입니다.',
                  style: TextStyle(
                      fontSize: 12,
                      color: Colors.red,
                      fontWeight: FontWeight.w600),
                ),
              SizedBox(
                height: 15,
              ),
              InputForm(
                  name: '비밀번호',
                  hintTxt: '8~20자의 영문, 숫자, 특수문자 조합',
                  needBtn: false),
              if (!isCorrectFormat)
                Text(
                  '8~20자의 영문, 숫자, 특수문자를 모두 포함해주세요.',
                  style: TextStyle(
                      fontSize: 12,
                      color: Colors.red,
                      fontWeight: FontWeight.w600),
                ),
              SizedBox(
                height: 15,
              ),
              InputForm(
                name: '비밀번호 확인',
                hintTxt: '비밀번호를 한 번 더 입력해주세요.',
                needBtn: false,
              ),
              if (!isCorrectPw)
                Text(
                  '비밀번호가 일치하지 않습니다.',
                  style: TextStyle(
                      fontSize: 12,
                      color: Colors.red,
                      fontWeight: FontWeight.w600),
                ),
              SizedBox(
                height: 35,
              ),
              SizedBox(
                width: 200,
                child: ButtonForm(
                  // 버튼 비활성화 추가하기
                  btnName: '제출',
                  buttonColor: Color(0xFF424242),
                  clickedFunc: submitBtn,
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
  final String name, hintTxt, btnName;
  final bool needBtn;
  final VoidCallback? clickedFunc;

  InputForm(
      {required this.name,
      required this.hintTxt,
      required this.needBtn,
      this.btnName = "",
      this.clickedFunc}) {
    print(clickedFunc);
  }

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
            child: ButtonForm(btnName: btnName, clickedFunc: clickedFunc),
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
  ButtonForm({
    super.key,
    required this.btnName,
    this.buttonColor = const Color(0xFFD97D6C),
    this.isTextBlack = false,
    required this.clickedFunc,
  }) {
    print(clickedFunc);
  }

  final String btnName;
  final Color buttonColor;
  final bool isTextBlack;
  final VoidCallback? clickedFunc;

  @override
  Widget build(BuildContext context) {
    return ElevatedButton(
      style: ElevatedButton.styleFrom(
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(12.0), // 버튼 테두리 둥글게
        ),
        padding:
            EdgeInsets.symmetric(vertical: 16.5, horizontal: 10), // 버튼 크기 설정
        backgroundColor: buttonColor, // 버튼 배경색
      ),
      onPressed: clickedFunc,
      child: Center(
        child: Text(
          btnName,
          style: TextStyle(
            fontSize: 15.0,
            color: isTextBlack ? Colors.black : Colors.white,
          ),
        ),
      ),
    );
  }
}
