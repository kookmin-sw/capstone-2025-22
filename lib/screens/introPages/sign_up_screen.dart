import 'package:capstone_2025/screens/introPages/login_screen.dart';
import 'package:capstone_2025/screens/introPages/widgets/intro_page_header.dart';
import 'package:capstone_2025/services/storage_service.dart';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'dart:async';

class SignUpScreen extends StatefulWidget {
  const SignUpScreen({super.key});

  @override
  State<SignUpScreen> createState() => _SignUpScreenState();
}

class _SignUpScreenState extends State<SignUpScreen> {
  final _formKey = GlobalKey<FormState>();

  final TextEditingController idController = TextEditingController();
  final TextEditingController numController = TextEditingController();
  final TextEditingController nameController = TextEditingController();
  final TextEditingController pwController = TextEditingController();
  final TextEditingController pwConfirmController = TextEditingController();

  String? _idErrorMessage;
  String? _nameErrorMessage;
  String? _pwErrorMessage;
  String? _pwConfirmErrorMessage;

  bool isEmailValidate = false;
  bool isAuthCodeRight = false;
  bool isNameValidate = false;
  bool isPwValidate = false;
  bool isPwCorrect = false;
  bool submitErr = false;

  // 🔹 타이머 관련 변수 추가
  int _remainingTime = 180; // 3분 (180초)
  bool _isTimerRunning = false;
  Timer? _timer;

  Future<bool> handleHTTP(
      String endpoint, Map<String, dynamic> queryParam) async {
    try {
      print("GET 요청 시작 --");

      final uri = Uri.http(
        "10.0.2.2:28080", // 서버 주소 (에뮬레이터용)
        endpoint, // 엔드포인트
        queryParam,
      );

      final response = await http.get(
        uri,
        headers: {
          "Accept": "application/json",
        },
      );

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        print("서버 응답: $data");

        return true;
      } else {
        print("서버 오류: ${response.statusCode} - ${response.body}");
        setState(() {
          _idErrorMessage = "서버 오류 발생: ${response.statusCode}";
        });
        return false;
      }
    } catch (error) {
      print("API 요청 실패: $error");
      setState(() {
        _idErrorMessage = "네트워크 오류 발생";
      });
      return false;
    }
  }

  Future<void> emailAuth() async {
    setState(() {
      String value = idController.text;

      if (value.isEmpty) {
        _idErrorMessage = "이메일을 입력해주세요.";
        return;
      }

      if (!RegExp(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
          .hasMatch(value)) {
        _idErrorMessage = "올바른 이메일 형식이 아닙니다.";
        return;
      }

      _idErrorMessage = null; // 이메일이 올바른 경우 에러 메시지 제거
      storage.write(key: "email", value: value);
    });

    // JSON 데이터 정의
    final Map<String, String> queryParam = {
      "email": idController.text,
    };
    isEmailValidate = await handleHTTP("/verification/auth-codes", queryParam);
  }

  Future<void> checkAuthCode() async {
    // JSON 데이터 정의
    final Map<String, dynamic> queryParam = {
      "email": storage.read(key: "email"),
      "authCode": numController.text,
    };
    isAuthCodeRight =
        await handleHTTP("/verification/auth-codes/check", queryParam);
  }

  Future<void> nameAuth() async {
    setState(() {
      String value = nameController.text;
      if (value.isEmpty) {
        _nameErrorMessage = "닉네임을 입력해주세요.";
        return;
      } else if (value.length < 2 || value.length > 8) {
        _nameErrorMessage = "닉네임은 2~8자여야 합니다.";
        return;
      }
      _nameErrorMessage = null;
    });
    // JSON 데이터 정의
    final Map<String, dynamic> queryParam = {
      "nickname": nameController.text,
    };
    isNameValidate = await handleHTTP("/verification/nicknames", queryParam);
  }

  void passwordAuth() {
    setState(() {
      String value = pwController.text;
      if (value.isEmpty) {
        _pwErrorMessage = "비밀번호를 입력해주세요.";
      } else if (!RegExp(
              r"^(?=.*[A-Za-z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,20}$")
          .hasMatch(value)) {
        _pwErrorMessage = "8~20자의 영문, 숫자, 특수문자 조합이어야 합니다.";
      } else {
        _pwErrorMessage = null;
        isPwValidate = true;
      }
    });
  }

  void passwordConfirmAuth() {
    setState(() {
      if (pwConfirmController.text.isEmpty) {
        _pwConfirmErrorMessage = "비밀번호를 한 번 더 입력해주세요.";
      } else if (pwConfirmController.text != pwController.text) {
        _pwConfirmErrorMessage = "비밀번호가 일치하지 않습니다.";
      } else {
        _pwConfirmErrorMessage = null;
        isPwCorrect = true;
      }
    });
  }

  // 🔹 타이머 시작 함수 추가
  void startTimer() {
    if (_timer != null) {
      _timer!.cancel(); // 기존 타이머가 있으면 취소
    }

    setState(() {
      _remainingTime = 180; // 3분
      _isTimerRunning = true;
    });

    _timer = Timer.periodic(Duration(seconds: 1), (timer) {
      if (_remainingTime > 0) {
        setState(() {
          _remainingTime--;
        });
      } else {
        timer.cancel();
        setState(() {
          _isTimerRunning = false;
        });
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Theme.of(context).scaffoldBackgroundColor,
      body: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 30, vertical: 10),
        child: SingleChildScrollView(
          child: Column(
            children: [
              introPageHeader(
                title: '회원가입',
                targetPage: LoginScreen(),
              ),
              Form(
                key: _formKey,
                child: Column(
                  children: [
                    inputForm(
                      tag: "아이디",
                      hintText: '본인인증을 위한 이메일 주소를 입력해주세요.',
                      onChangedFunc: (value) {
                        setState(() {
                          _idErrorMessage = null;
                        });
                      },
                      needBtn: true,
                      btnName: "전송",
                      btnFunc: emailAuth,
                      controller: idController,
                      errorMessage: _idErrorMessage,
                    ),
                    SizedBox(height: 25),
                    // 🔹 인증번호 입력칸 오른쪽에 타이머 추가
                    inputForm(
                      tag: "인증번호",
                      hintText: '인증번호 6자리를 입력해주세요.',
                      onChangedFunc: (value) {},
                      needBtn: true,
                      btnName: "확인",
                      controller: numController,
                      btnFunc: () {},
                      additionalWidget: _isTimerRunning
                          ? Text(
                              "${_remainingTime ~/ 60}:${(_remainingTime % 60).toString().padLeft(2, '0')}",
                              style: TextStyle(
                                  color: Colors.red,
                                  fontSize: 16,
                                  fontWeight: FontWeight.bold),
                            )
                          : null,
                    ),
                    SizedBox(height: 25),
                    inputForm(
                        tag: "닉네임",
                        hintText: '2~8자',
                        onChangedFunc: (value) {
                          nameAuth();
                        },
                        needBtn: true,
                        btnName: "중복확인",
                        controller: nameController,
                        errorMessage: _nameErrorMessage,
                        btnFunc: () {
                          setState(() {
                            isNameValidate = true; // 닉네임 중복 확인 버튼 함수
                          });
                        }),
                    SizedBox(height: 25),
                    inputForm(
                      tag: "비밀번호",
                      hintText: '8~20자의 영문, 숫자, 특수문자 조합',
                      onChangedFunc: (value) {
                        passwordAuth();
                      },
                      needBtn: false,
                      controller: pwController,
                      errorMessage: _pwErrorMessage,
                    ),
                    SizedBox(height: 25),
                    inputForm(
                      tag: "비밀번호 확인",
                      hintText: '비밀번호를 한 번 더 입력해주세요.',
                      onChangedFunc: (value) {
                        passwordConfirmAuth();
                      },
                      needBtn: false,
                      controller: pwConfirmController,
                      errorMessage: _pwConfirmErrorMessage,
                    ),
                    SizedBox(height: 40),
                    SizedBox(
                      width: 300,
                      height: 60,
                      child: ButtonForm(
                        btnName: "제출",
                        buttonColor: Color(0xFF424242),
                        clickedFunc: () {
                          if (isEmailValidate &&
                              isNameValidate &&
                              isAuthCodeRight &&
                              isPwCorrect &&
                              isPwValidate) {
                            submitErr = false;
                            Navigator.of(context).pushReplacement(
                                MaterialPageRoute(
                                    builder: (_) => LoginScreen()));
                          } else {
                            setState(() {
                              submitErr = true;
                            });
                          }
                        },
                      ),
                    ),
                    if (submitErr)
                      Padding(
                        padding: const EdgeInsets.symmetric(vertical: 10),
                        child: Text(
                          "입력된 정보를 다시 확인해주세요. 필수 항목이 비어있거나 조건을 만족하지 않았습니다.",
                          style: TextStyle(
                            color: Colors.red,
                            fontSize: 14,
                            fontWeight: FontWeight.w500,
                          ),
                        ),
                      ),
                    SizedBox(height: 30),
                  ],
                ),
              )
            ],
          ),
        ),
      ),
    );
  }
}

Row inputForm({
  required String tag,
  double fontSize = 18,
  required String hintText,
  required Function(String) onChangedFunc,
  required bool needBtn,
  String btnName = 'null',
  Widget? additionalWidget, // 추가 UI 요소 (예: 타이머)
  VoidCallback? btnFunc,
  TextEditingController? controller,
  String? errorMessage,
}) {
  return Row(
    mainAxisAlignment: MainAxisAlignment.center,
    children: [
      Expanded(flex: 8, child: SizedBox()),
      Expanded(
        flex: 5,
        child: Text(
          tag,
          style: TextStyle(
            fontSize: fontSize,
            fontWeight: FontWeight.w600,
            color: Colors.black38,
          ),
          textAlign: TextAlign.end,
        ),
      ),
      Expanded(flex: 1, child: SizedBox()),
      Expanded(
        flex: 19,
        child: TextFormField(
          controller: controller,
          obscureText: tag.contains("비밀번호"),
          decoration: InputDecoration(
            hintText: hintText,
            errorText: errorMessage,
            filled: false,
            fillColor: Colors.white,
            enabledBorder: OutlineInputBorder(
              borderRadius: BorderRadius.circular(15),
              borderSide: BorderSide(color: Colors.grey.shade300, width: 1),
            ),
            focusedBorder: OutlineInputBorder(
              borderRadius: BorderRadius.circular(15),
              borderSide: BorderSide(color: Colors.grey, width: 1.5),
            ),
          ),
          onChanged: (value) {
            onChangedFunc(value);
          },
        ),
      ),
      Expanded(flex: 1, child: SizedBox()),
      if (needBtn)
        Expanded(
          flex: 4,
          child: ButtonForm(btnName: btnName, clickedFunc: btnFunc),
        ),
      Expanded(flex: needBtn ? 7 : 11, child: SizedBox()),
    ],
  );
}

class ButtonForm extends StatelessWidget {
  const ButtonForm({
    super.key,
    required this.btnName,
    this.buttonColor = const Color(0xFFD97D6C),
    this.isTextBlack = false,
    this.clickedFunc,
  });

  final String btnName;
  final Color buttonColor;
  final bool isTextBlack;
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
