import 'package:capstone_2025/screens/introPages/login_screen.dart';
import 'package:capstone_2025/screens/introPages/login_screen_google.dart';
import 'package:capstone_2025/screens/introPages/widgets/intro_page_header.dart';
import 'package:capstone_2025/services/storage_service.dart';
import 'package:flutter/foundation.dart';
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
  String? _codeErrorMessage;
  String? _nameErrorMessage;
  String? _pwErrorMessage;
  String? _pwConfirmErrorMessage;
  String errMessage = " ";

  bool isEmailValidate = false;
  bool isAuthCodeRight = false;
  bool isNameValidate = false;
  bool isPwValidate = false;
  bool isPwCorrect = false;
  bool submitErr = false;

  bool isEmailButtonEnabled = true; // 이메일 전송 버튼 활성화 여부
  bool isAuthButtonEnabled = false; // 인증번호 확인 버튼 활성화 여부
  bool isNameButtonEnabled = true; // 닉네임 중복확인 버튼 활성화 여부

  // 🔹 타이머 관련 변수 추가
  late Timer _timer;
  int _timeRemaining = 180; // 남은 시간 3분 (초 단위)
  bool _isTimerRunning = false; // 타이머가 실행 중인지 여부

  Future<Map<String, dynamic>> handleHTTP(
      String endpoint, Map<String, dynamic> queryParam) async {
    try {
      print("GET 요청 시작 --");

      final uri = Uri.http(
        "10.0.2.2:28080", // 서버 주소 (에뮬레이터용)
        // "192.168.219.108:28080", // 서버 주소 (실제 기기용- 아이폰)
        endpoint, // 엔드포인트
        queryParam,
      );

      final response = await http.get(
        uri,
        headers: {
          "Accept": "application/json",
        },
      );
      print("_________________");
      print(response.body);
      print("_________________");

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        print("서버 응답: $data");

        setState(() {
          errMessage = " ";
        });
        return data;
      } else {
        print("서버 오류: ${response.statusCode} - ${response.body}");
        setState(() {
          errMessage = "서버 오류 발생: ${response.statusCode}";
        });
        return {};
      }
    } catch (error) {
      print("API 요청 실패: $error");
      setState(() {
        errMessage = "네트워크 오류 발생";
      });
      return {};
    }
  }

  Future<void> emailAuth() async {
    String value = idController.text;
    setState(() {
      isEmailButtonEnabled = false;
      _idErrorMessage = null; // 기존 오류 메시지 초기화
    });

    setState(() {
      if (value.isEmpty) {
        isEmailButtonEnabled = false;
        _idErrorMessage = "이메일을 입력해주세요.";
        return;
      }

      if (!RegExp(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
          .hasMatch(value)) {
        isEmailButtonEnabled = false;
        _idErrorMessage = "올바른 이메일 형식이 아닙니다.";
        return;
      }
    });

    // JSON 데이터 정의
    final Map<String, String> queryParam = {
      "email": idController.text,
    };
    Map<String, dynamic> resData =
        await handleHTTP("/verification/auth-codes", queryParam);
    if (resData == {}) {
      return;
    }
    if (resData["body"] == "valid") {
      setState(() {
        isEmailValidate = true;
        isAuthButtonEnabled = true;
        _idErrorMessage = "인증번호가 전송되었습니다.";
        storage.write(key: "email", value: value);
        _startTimer(); // 타이머 시작
      });
      return;
    }
    if (resData["body"] == "invalid") {
      _idErrorMessage = "이미 가입된 이메일 주소입니다."; // 이메일이 올바른 경우 메시지 출력
      return;
    }
  }

  Future<void> checkAuthCode() async {
    // JSON 데이터 정의
    final Map<String, dynamic> queryParam = {
      "email": await storage.read(key: "email"),
      "authCode": numController.text,
    };
    Map<String, dynamic> resData =
        await handleHTTP("/verification/auth-codes/check", queryParam);

    if (resData == {}) {
      _codeErrorMessage = "error";
      return;
    }

    if (resData["body"] == "invalid") {
      print(resData["body"]);
      _codeErrorMessage = "인증번호가 틀렸습니다.";
      return;
    } else {
      print(resData["body"]);
      // storage.write(key: "emailToken", value: resData["body"]["emailToken"]);
      isAuthCodeRight = true;
      _timer.cancel();
      _isTimerRunning = false;
      _codeErrorMessage = null;
      _idErrorMessage = null;
      isAuthButtonEnabled = false;
      return;
    }
  }

  Future<void> nameAuth() async {
    // JSON 데이터 정의
    final Map<String, dynamic> queryParam = {
      "nickname": nameController.text,
    };
    Map<String, dynamic> resData =
        await handleHTTP("/verification/nicknames", queryParam);
    if (resData == {}) {
      isNameValidate = false;
      _nameErrorMessage = "error";
      return;
    }
    if (resData["body"] == "valid") {
      isNameValidate = true;
      _nameErrorMessage = null;
      return;
    }
    if (resData["body"] == "invalid") {
      isNameValidate = false;

      _nameErrorMessage = "이미 사용 중인 닉네임입니다.";
      return;
    }
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

  Future<Map<String, dynamic>?> sendUserDataToServer(
      String email, String password, String nickname) async {
    final Map<String, dynamic> requestBody = {
      "email": email,
      "password": password,
      "nickname": nickname,
    };

    try {
      // http post
      final response =
          // "http://192.168.219.108:28080/auth/signup// 에뮬레이터용
          await http.post(Uri.parse("http://10.0.2.2:28080/auth/signup"),
              headers: {
                "Content-Type": "application/json",
                "Accept": "application/json",
              },
              body: jsonEncode(requestBody));

      if (response.statusCode == 200) {
        print("회원가입 성공!");
        final data = jsonDecode(response.body);
        return data; // 사용자 정보 반환
      } 
      if (response.statusCode == 409) {
        errMessage = "이미 가입된 이메일 주소입니다.";
        return null;
        } else {
        print("서버 오류: ${response.statusCode} - ${response.body}");
        return null;
      }
    } catch (error) {
      print("API 요청 실패 : $error");
    }
  }

// Response 받은 정보들 저장하는 함수
  Future<void> saveUserInfo(Map<String, dynamic> userInfo) async {
    await storage.write(key: "user_email", value: userInfo["email"]);
    await storage.write(key: "user_name", value: userInfo["name"]);
  }

  void signUpComplete() async {
    if (isEmailValidate &&
        isNameValidate &&
        isAuthCodeRight &&
        isPwCorrect &&
        isPwValidate) {
      submitErr = false;

      var userInfo = await sendUserDataToServer(
          idController.text, pwController.text, nameController.text);

      if (userInfo != null) {
        await saveUserInfo(userInfo);
        if (mounted) {
          Navigator.of(context).pushReplacement(
            MaterialPageRoute(builder: (_) => LoginScreenGoogle()),
          );
        }
      }
    } else {
      setState(() {
        submitErr = true;
      });
    }
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
        print("타이머 종료!"); // 타이머 종료 로그 추가
        setState(() {
          _isTimerRunning = false;
          isEmailButtonEnabled = true;
        });
      }
    });
    if (!(_isTimerRunning && isAuthButtonEnabled)) {
      _codeErrorMessage = "인증번호가 만료되었습니다.";
      isEmailButtonEnabled = true;
    }
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
      backgroundColor: Theme.of(context).scaffoldBackgroundColor,
      body: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 30, vertical: 10),
        child: SingleChildScrollView(
          child: Column(
            children: [
              introPageHeader(
                title: '회원가입',
                targetPage: LoginScreenGoogle(),
              ),
              Text(
                errMessage,
                style: TextStyle(
                  color: Colors.red,
                  fontSize: 14,
                  fontWeight: FontWeight.w800,
                ),
              ),
              SizedBox(
                height: 5,
              ),
              Form(
                key: _formKey,
                child: Column(
                  children: [
                    inputForm(
                      tag: "아이디",
                      hintText: '본인인증을 위한 이메일 주소를 입력해주세요.',
                      onChangedFunc: (value) {
                        setState(
                          () {
                            _idErrorMessage = null;
                          },
                        );
                      },
                      needBtn: true,
                      btnName: "전송",
                      btnFunc: emailAuth,
                      controller: idController,
                      errorMessage: _idErrorMessage,
                      isEnabled: isEmailButtonEnabled,
                    ),
                    SizedBox(height: 25),
                    // 인증번호 입력칸 오른쪽에 타이머 추가
                    inputForm(
                        tag: "인증번호",
                        hintText: '인증번호 6자리를 입력해주세요.',
                        onChangedFunc: (value) {},
                        needBtn: true,
                        btnName: "확인",
                        controller: numController,
                        btnFunc: checkAuthCode,
                        errorMessage: _codeErrorMessage,
                        timerString: _isTimerRunning
                            ? _formatTime(_timeRemaining)
                            : null,
                        isEnabled: isAuthButtonEnabled),
                    SizedBox(
                      height: 5,
                    ),
                    if (isAuthCodeRight)
                      Text(
                        "인증되었습니다.",
                        style: TextStyle(
                          color: const Color.fromARGB(255, 12, 148, 16),
                          fontWeight: FontWeight.w600,
                          fontSize: 12,
                        ),
                      ),
                    SizedBox(height: isAuthCodeRight ? 20 : 25),
                    inputForm(
                      tag: "닉네임",
                      hintText: '2~8자',
                      onChangedFunc: (value) {
                        setState(() {
                          String value = nameController.text;
                          if (value.isEmpty) {
                            isNameButtonEnabled = false;
                            _nameErrorMessage = "닉네임을 입력해주세요.";
                            return;
                          } else if (value.length < 2 || value.length > 8) {
                            isNameButtonEnabled = false;
                            _nameErrorMessage = "닉네임은 2~8자여야 합니다.";
                            return;
                          }
                          _nameErrorMessage = null;
                          isNameButtonEnabled = true;
                        });
                      },
                      needBtn: true,
                      btnName: "중복확인",
                      controller: nameController,
                      errorMessage: _nameErrorMessage,
                      btnFunc: nameAuth,
                      isEnabled: isNameButtonEnabled,
                    ),
                    SizedBox(
                      height: 5,
                    ),
                    if (isNameValidate)
                      Text(
                        "사용 가능한 닉네임입니다.",
                        style: TextStyle(
                            color: const Color.fromARGB(255, 12, 148, 16),
                            fontWeight: FontWeight.w600,
                            fontSize: 12),
                      ),
                    SizedBox(height: isAuthCodeRight ? 20 : 25),
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
                        clickedFunc: signUpComplete,
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
  String? timerString, // 추가 UI 요소 (예: 타이머)
  VoidCallback? btnFunc,
  TextEditingController? controller,
  String? errorMessage,
  bool isEnabled = true,
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
        child: Stack(
          alignment: Alignment.centerRight, // 타이머를 오른쪽 끝에 배치
          children: [
            TextFormField(
              controller: controller,
              obscureText: tag.contains("비밀번호"),
              textAlign: TextAlign.left, // 입력값을 왼쪽 정렬
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
            // 타이머 추가
            if (timerString != null)
              Positioned(
                right: 15, // 타이머를 오른쪽으로 배치
                child: Text(
                  timerString,
                  style: TextStyle(
                    fontSize: 16,
                    color: Colors.red,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
          ],
        ),
      ),
      Expanded(flex: 1, child: SizedBox()),
      if (needBtn)
        Expanded(
          flex: 4,
          child: ButtonForm(
            btnName: btnName,
            clickedFunc: btnFunc,
            isEnabled: isEnabled,
          ),
        ),
      Expanded(flex: needBtn ? 7 : 11, child: SizedBox()),
    ],
  );
}

class ButtonForm extends StatelessWidget {
  ButtonForm({
    super.key,
    required this.btnName,
    this.buttonColor = const Color(0xFFD97D6C),
    this.isTextBlack = false,
    this.clickedFunc,
    this.isEnabled = true,
  });

  final String btnName;
  final Color buttonColor;
  final bool isTextBlack;
  final VoidCallback? clickedFunc;
  bool isEnabled = true;

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
      onPressed: isEnabled ? clickedFunc : null,
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
