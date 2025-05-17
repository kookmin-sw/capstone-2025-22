import 'dart:convert';
import 'package:capstone_2025/widgets/complete_dialog.dart';
import 'package:flutter_screenutil/flutter_screenutil.dart';
import 'package:capstone_2025/screens/introPages/login_screen_google.dart';
import 'package:capstone_2025/screens/introPages/widgets/intro_page_header.dart';
import 'package:capstone_2025/services/api_func.dart';
import 'package:capstone_2025/services/storage_service.dart';
import 'package:flutter/material.dart';
import 'dart:async';

class SignUpScreen extends StatefulWidget {
  // 회원가입 페이지
  const SignUpScreen({super.key});

  @override
  State<SignUpScreen> createState() => _SignUpScreenState();
}

class _SignUpScreenState extends State<SignUpScreen> {
  final _formKey = GlobalKey<FormState>();

  // TextField Controller들
  final TextEditingController idController = TextEditingController();
  final TextEditingController numController = TextEditingController();
  final TextEditingController nameController = TextEditingController();
  final TextEditingController pwController = TextEditingController();
  final TextEditingController pwConfirmController = TextEditingController();

  // Error Message들
  String? _idErrorMessage;
  String? _codeErrorMessage; // 인증번호
  String? _nameErrorMessage;
  String? _pwErrorMessage;
  String? _pwConfirmErrorMessage;
  String errMessage = " "; // 기타 (최종 에러 메세지)

  // 각 field 유효성 변수
  bool isEmailValidate = false;
  bool isAuthCodeRight = false;
  bool isNameValidate = false;
  bool isPwValidate = false;
  bool isPwCorrect = false;
  bool submitErr = false;

  bool isEmailButtonEnabled = true; // 이메일 전송 버튼 활성화 여부
  bool isAuthButtonEnabled = false; // 인증번호 확인 버튼 활성화 여부
  bool isNameButtonEnabled = true; // 닉네임 중복확인 버튼 활성화 여부

  bool isLoading = false; // 로딩 중 여부

  // 타이머 관련 변수 추가
  late Timer _timer;
  int _timeRemaining = 180; // 남은 시간 3분 (초 단위)
  bool _isTimerRunning = false; // 타이머가 실행 중인지 여부

  @override
  void initState() {
    super.initState();
    // 타이머 시작
    _timer = Timer.periodic(const Duration(seconds: 1), (timer) {
      // 아래 2번 mounted 검사도 함께 적용
      if (!mounted) {
        timer.cancel();
        return;
      }

      if (_timeRemaining > 0) {
        setState(() {
          _timeRemaining--;
        });
      } else {
        timer.cancel();
      }
    });
  }

  @override
  void dispose() {
    // 화면이 사라질 때 타이머 종료
    _timer.cancel();
    super.dispose();
  }

  // 이메일 인증 및 인증코드 전송 로직 함수
  Future<void> emailAuth() async {
    String value = idController.text; // 입력된 아이디(이메일) 받아오기
    showLoadingDialog(context); // 로딩창 표시
    setState(() {
      isEmailButtonEnabled = false; // 인증번호 전송 버튼 중복 클릭 방지
      _idErrorMessage = null; // 기존 오류 메시지 초기화
      _codeErrorMessage = null; // 기존 오류 메시지 초기화
      isLoading = true; // 로딩 중 변수 업데이트
    });

    setState(() {
      if (value.isEmpty) {
        // 이메일을 입력하지 않았을 때
        // 모달이 떠 있는지 확인 후 닫기 (화면 전체 pop 방지)
        if (Navigator.canPop(context)) {
          Navigator.pop(context); // 모달만 닫기
        } // 로딩창 닫기
        isEmailButtonEnabled = true;
        _idErrorMessage = "이메일을 입력해주세요.";
        return;
      }
      // 이메일 정규 표현식
      if (!RegExp(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
          .hasMatch(value)) {
        // 모달이 떠 있는지 확인 후 닫기 (화면 전체 pop 방지)
        if (Navigator.canPop(context)) {
          Navigator.pop(context); // 모달만 닫기
        } // 로딩창 닫기
        isEmailButtonEnabled = true;
        _idErrorMessage = "올바른 이메일 형식이 아닙니다.";
        return;
      }
    });
    // JSON 데이터 정의
    final Map<String, String> queryParam = {
      // API를 통해 전달할 param
      "email": idController.text,
    };
    // 이메일 존재 여부 확인
    Map<String, dynamic> resData =
        await getHTTP("/verification/emails", queryParam);
    if (resData["body"] == "invalid") {
      // 이메일이 이미 존재할 경우
      _idErrorMessage = "이미 가입된 이메일 주소입니다."; // 이메일이 올바른 경우 메시지 출력
      setState(() {
        isEmailButtonEnabled = true; // 인증번호 전송 버튼 활성화
        // 모달이 떠 있는지 확인 후 닫기 (화면 전체 pop 방지)
        if (Navigator.canPop(context)) {
          Navigator.pop(context); // 모달만 닫기
        } // 로딩창 닫기
      });
      return;
    }
    if (resData['errMessage'] != null) {
      // error 발생 시
      _idErrorMessage = resData['errMessage'];
      // 모달이 떠 있는지 확인 후 닫기 (화면 전체 pop 방지)
      if (Navigator.canPop(context)) {
        Navigator.pop(context); // 모달만 닫기
      } // 로딩창 닫기
      return;
    }

    // 인증 코드 전송
    resData = await getHTTP("/verification/auth-codes", queryParam);
    if (resData["body"] == "valid") {
      // 정상적인 경우
      setState(() {
        // 모달이 떠 있는지 확인 후 닫기 (화면 전체 pop 방지)
        if (Navigator.canPop(context)) {
          Navigator.pop(context); // 모달만 닫기
        } // 로딩창 닫기
        isEmailValidate = true; // 유효성 변수 업데이트
        isAuthButtonEnabled = true; // 인증번호 확인 버튼 활성화
        isLoading = false; // 로딩 중 변수 업데이트
        _idErrorMessage = "인증번호가 전송되었습니다.";
        storage.write(key: "email", value: value); // storage에 이메일 저장
        _startTimer(); // 타이머 시작
      });
      return;
    }
    if (resData['errMessage'] != null) {
      // 모달이 떠 있는지 확인 후 닫기 (화면 전체 pop 방지)
      if (Navigator.canPop(context)) {
        Navigator.pop(context); // 모달만 닫기
      } // 로딩창 닫기
      _idErrorMessage = resData['errMessage'];
      return;
    }
    if (resData["body"] == "invalid") {
      // 모달이 떠 있는지 확인 후 닫기 (화면 전체 pop 방지)
      if (Navigator.canPop(context)) {
        Navigator.pop(context); // 모달만 닫기
      } // 로딩창 닫기
      // 이미 유효할 경우
      _idErrorMessage = "이미 가입된 이메일 주소입니다."; // 이메일이 올바른 경우 메시지 출력
      return;
    }
  }

  // 인증 코드 확인 로직 함수
  Future<void> checkAuthCode() async {
    // JSON 데이터 정의
    final Map<String, dynamic> queryParam = {
      "email": await storage.read(key: "email"), // 저장된 이메일 가져오기
      "authCode": numController.text, // 입력된 인증 코드
    };
    Map<String, dynamic> resData = // API 호출
        await getHTTP("/verification/auth-codes/check", queryParam);

    if (resData["body"] == null) {
      // 인증 코드 틀렸을 때
      print(resData["body"]);
      _codeErrorMessage = "인증번호가 틀렸습니다.";
      return;
    }
    if (resData['errMessage'] != null) {
      _codeErrorMessage = "error";
      return;
    } else {
      // 인증 코드 맞았을 때
      print(resData["body"]);
      storage.write(
          key: "emailToken",
          value: resData["body"]["emailToken"]); // email token 저장
      isAuthCodeRight = true;
      _timer.cancel(); // 타이머 종료
      setState(() {
        _isTimerRunning = false; // 타이머 변수 설정
        _codeErrorMessage = null; // 인증번호 전송 알림 문구 삭제
        _idErrorMessage = null;
        isAuthButtonEnabled = false; // 인증번호 확인 버튼 비활성화
      });
      return;
    }
  }

  // 닉네임 유효성 검사 함수
  Future<void> nameAuth() async {
    // JSON 데이터 정의
    final Map<String, dynamic> queryParam = {
      "nickname": nameController.text,
    };
    Map<String, dynamic> resData =
        await getHTTP("/verification/nicknames", queryParam);

    setState(() {
      if (resData["body"] == "valid") {
        // 닉네임 사용 가능
        isNameValidate = true;
        _nameErrorMessage = null;
        return;
      }
      if (resData["body"] == "invalid") {
        // 닉네임 중복
        isNameValidate = false;

        _nameErrorMessage = "이미 사용 중인 닉네임입니다.";
        return;
      }
      if (resData['errMessage'] != null) {
        isNameValidate = false;
        _nameErrorMessage = "error";
        return;
      }
    });
  }

// 비밀번호 유효성 검사 함수
  void passwordAuth() {
    setState(() {
      String value = pwController.text;
      if (value.isEmpty) {
        // 입력란 비어있을 때
        _pwErrorMessage = "비밀번호를 입력해주세요.";
      } else if (!RegExp(// 정규 표현식
              r"^(?=.*[A-Za-z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,20}$")
          .hasMatch(value)) {
        _pwErrorMessage = "8~20자의 영문, 숫자, 특수문자 조합이어야 합니다.";
      } else {
        // 정상적인 경우
        _pwErrorMessage = null;
        isPwValidate = true;
      }
    });
  }

  // 비밀번호 확인 함수
  void passwordConfirmAuth() {
    setState(() {
      if (pwConfirmController.text.isEmpty) {
        _pwConfirmErrorMessage = "비밀번호를 한 번 더 입력해주세요.";
      } else if (pwConfirmController.text != pwController.text) {
        _pwConfirmErrorMessage = "비밀번호가 일치하지 않습니다.";
      } else {
        // 정상적인 경우
        _pwConfirmErrorMessage = null;
        isPwCorrect = true;
      }
    });
  }

// Response 받은 정보들 저장하는 함수
  Future<void> saveUserInfo(Map<String, dynamic> userInfo) async {
    // secure Storage에 저장
    await storage.write(key: "user_email", value: userInfo["email"]);
    await storage.write(
      key: 'user_name',
      value: utf8.decode(userInfo["name"].toString().codeUnits),
    );
  }

  // 회원가입 완료 여부 확인 함수
  void signUpComplete() async {
    if (isEmailValidate &&
        isNameValidate &&
        isAuthCodeRight &&
        isPwCorrect &&
        isPwValidate &&
        !_isTimerRunning) {
      submitErr = false;
      showLoadingDialog(context);

      final Map<String, dynamic> requestBody = {
        "email": idController.text,
        "password": pwController.text,
        "nickname": nameController.text,
      };

      var userInfo = await postHTTP("/auth/signup", requestBody);

      if (Navigator.canPop(context)) {
        Navigator.pop(context); // 로딩창 닫기
      }

      if (userInfo['errMessage'] == null) {
        await saveUserInfo(userInfo);
        if (mounted) {
          await showDialog(
            context: context,
            builder: (BuildContext context) {
              return CompleteDialog(
                mainText: "회원가입이 완료되었습니다.",
                subText: "지금 바로 로그인해보세요!",
                onClose: () {
                  Navigator.of(context).pop();
                  Navigator.of(context).pushReplacement(
                    MaterialPageRoute(builder: (_) => LoginScreenGoogle()),
                  );
                },
              );
            },
          );
        }
      } else {
        setState(() {
          submitErr = true;
          errMessage = userInfo['errMessage'] ??
              "입력된 정보를 다시 확인해주세요. 필수 항목이 비어있거나 조건을 만족하지 않았습니다.";
        });
      }
    } else {
      if (Navigator.canPop(context)) {
        Navigator.pop(context); // 로딩창 닫기
      }
      setState(() {
        submitErr = true;
        errMessage = "입력된 정보를 다시 확인해주세요. 필수 항목이 비어있거나 조건을 만족하지 않았습니다.";
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
      if (!mounted) {
        timer.cancel();
        return;
      }

      if (_timeRemaining > 0) {
        // 1초마다 _timeRemaining 값을 1씩 감소
        setState(() {
          _timeRemaining--;
        });
      } else {
        // 시간이 0이 되면 _isTimerRunning을 false로 변경 (타이머 중지)
        timer.cancel();
        print("타이머 종료!"); // 타이머 종료 로그 추가
        setState(() {
          _isTimerRunning = false;
          isEmailButtonEnabled = true; // 시간 만료 - 인증번호 재전송 가능하게 만듦
          isAuthButtonEnabled = false; // 시간 만료 - 인증번호 확인 버튼 비활성화
          _codeErrorMessage = "인증번호가 만료되었습니다.";
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

  void showLoadingDialog(BuildContext context) {
    showDialog(
      context: context,
      barrierDismissible: false, // 사용자가 닫을 수 없도록 설정
      builder: (context) {
        return Dialog(
          backgroundColor: Colors.transparent,
          child: Center(
            child: CircularProgressIndicator(),
          ),
        );
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      resizeToAvoidBottomInset: true, // 키보드에 의해 UI 밀림 방지
      backgroundColor: Theme.of(context).scaffoldBackgroundColor,
      body: SingleChildScrollView(
        // 스크롤 가능하도록 설정
        physics: ClampingScrollPhysics(), // 스크롤 효과
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            introPageHeader(
              title: '회원가입',
              targetPage: LoginScreenGoogle(), // 뒤로가기 버튼 대상
            ),
            SizedBox(
              height: 35.h,
            ),
            Form(
              key: _formKey,
              child: Column(
                children: [
                  inputForm(
                    // 아이디 입력란
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
                  SizedBox(height: 25.h),
                  // 인증번호 입력칸 오른쪽에 타이머 추가
                  inputForm(
                      // 인증번호 입력란
                      tag: "인증번호",
                      hintText: '인증번호 6자리를 입력해주세요.',
                      onChangedFunc: (value) {},
                      needBtn: true,
                      btnName: "확인",
                      controller: numController,
                      btnFunc: checkAuthCode,
                      errorMessage: _codeErrorMessage,
                      timerString:
                          _isTimerRunning ? _formatTime(_timeRemaining) : null,
                      isEnabled: isAuthButtonEnabled),
                  SizedBox(
                    height: 5.h,
                  ),
                  if (isAuthCodeRight) // 인증번호 확인 완료
                    Text(
                      "인증되었습니다.",
                      style: TextStyle(
                        color: const Color.fromARGB(255, 12, 148, 16),
                        fontWeight: FontWeight.w600,
                        fontSize: 4.8.sp,
                      ),
                    ),
                  SizedBox(
                      height: isAuthCodeRight
                          ? 20.h
                          : 25.h), // 인증번호 메세지 유무에 따라 사이즈 조정
                  inputForm(
                    // 닉네임 입력란
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
                    height: 5.h,
                  ),
                  if (isNameValidate)
                    Text(
                      "사용 가능한 닉네임입니다.",
                      style: TextStyle(
                          color: const Color.fromARGB(255, 12, 148, 16),
                          fontWeight: FontWeight.w600,
                          fontSize: 4.8.sp),
                    ),
                  SizedBox(height: isAuthCodeRight ? 20 : 25),
                  inputForm(
                    // 비밀번호 입력란
                    tag: "비밀번호",
                    hintText: '8~20자의 영문, 숫자, 특수문자 조합',
                    onChangedFunc: (value) {
                      passwordAuth();
                    },
                    needBtn: false,
                    controller: pwController,
                    errorMessage: _pwErrorMessage,
                  ),
                  SizedBox(height: 25.h),
                  inputForm(
                    // 비밀번호 확인 입력란
                    tag: "비밀번호 확인",
                    hintText: '비밀번호를 한 번 더 입력해주세요.',
                    onChangedFunc: (value) {
                      passwordConfirmAuth();
                    },
                    needBtn: false,
                    controller: pwConfirmController,
                    errorMessage: _pwConfirmErrorMessage,
                  ),
                  SizedBox(height: 45.h),
                  SizedBox(
                    // 제출 버튼
                    width: 340.h,
                    height: 70.h,
                    child: ButtonForm(
                      btnName: "제출",
                      buttonColor: Color(0xFF424242),
                      clickedFunc: signUpComplete,
                    ),
                  ),
                  Padding(
                    // 에러 메세지
                    padding: EdgeInsets.symmetric(vertical: 10.h),
                    child: Text(
                      errMessage,
                      style: TextStyle(
                        color: Colors.red,
                        fontSize: 4.8.sp,
                        fontWeight: FontWeight.w500,
                      ),
                    ),
                  ),
                  SizedBox(height: 30.h),
                ],
              ),
            )
          ],
        ),
      ),
    );
  }
}

// 텍스트 입력 폼
Row inputForm({
  required String tag, // 입력폼 태그
  double? fontSize,
  required String hintText, // 입력 내용 조건문
  required Function(String) onChangedFunc, // 연결 함수
  required bool needBtn, // 버튼 유무
  String btnName = 'null', // 버튼 이름
  String? timerString, // 추가 UI 요소 (예: 타이머)
  VoidCallback? btnFunc, // 버튼 연결 함수
  TextEditingController? controller, //텍스트 컨트롤러
  String? errorMessage, // 에러메세지
  bool isEnabled = true, // 버튼 사용 가능 유무
}) {
  return Row(
    mainAxisAlignment: MainAxisAlignment.center,
    children: [
      Expanded(flex: 8, child: SizedBox()),
      Expanded(
        // 태그
        flex: 5,
        child: Text(
          tag,
          style: TextStyle(
            fontSize: fontSize ?? 6.sp,
            fontWeight: FontWeight.w600,
            color: Colors.black38,
          ),
          textAlign: TextAlign.end,
        ),
      ),
      Expanded(flex: 1, child: SizedBox()),
      Expanded(
        // 입력란
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
                right: 10.w, // 타이머를 오른쪽으로 배치
                child: Text(
                  timerString,
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
      Expanded(flex: 1, child: SizedBox()),
      if (needBtn) // 버튼
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
        padding: EdgeInsets.symmetric(vertical: 16.5.h, horizontal: 6.w),
        backgroundColor: buttonColor,
      ),
      onPressed: isEnabled ? clickedFunc : null,
      child: Center(
        child: Text(
          btnName,
          style: TextStyle(
            fontSize: 5.5.sp,
            color: isTextBlack ? Colors.black : Colors.white,
          ),
        ),
      ),
    );
  }
}
