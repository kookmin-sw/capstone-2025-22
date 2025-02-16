import 'package:capstone_2025/screens/introPages/find_pw_screen.dart';
import 'package:capstone_2025/screens/introPages/widgets/intro_page_header.dart';
import 'package:flutter/material.dart';
import 'dart:convert';
import 'package:http/http.dart' as http;

// 사용자가 본인 인증 완료 후 새로운 비밀번호를 설정하는 화면
class SetNewPwScreen extends StatefulWidget {
  const SetNewPwScreen({super.key});

  @override
  State<SetNewPwScreen> createState() => _SetNewPwScreenState();
}

class _SetNewPwScreenState extends State<SetNewPwScreen> {
  bool _isPasswordVisible = false;
  bool _isConfirmPasswordVisible = false; // 비밀번호 보기 상태 관리
  final _formKey = GlobalKey<FormState>(); // 폼 상태 관리
  final _passwordController = TextEditingController(); // 새 비밀번호 입력 제어 컨트롤러
  final _confirmPasswordController =
      TextEditingController(); // 새 비밀번호 확인 제어 컨트롤러

  // 비밀번호 유효성 검사
  String? _validatePassword(String? value) {
    if (value == null || value.isEmpty) {
      return '8~20자의 영문, 숫자, 특수문자를 모두 포함해주세요.';
    }
    if (!RegExp(
            r'^(?=.*[a-zA-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,20}$')
        .hasMatch(value)) {
      return '8~20자의 영문, 숫자, 특수문자를 모두 포함해주세요.';
    }
    return null;
  }

  String? _validateConfirmPassword(String? value) {
    if (value == null || value.isEmpty) {
      return '비밀번호를 다시 한 번 입력해주세요.';
    }
    if (value != _passwordController.text) {
      return '비밀번호가 일치하지 않습니다.';
    }
    return null;
  }

  // 비밀번호 재설정 API 호출
  Future<void> setNewPassword(
      String newPassword, String emailToken, String accessToken) async {
    final url = Uri.parse('https://10.0.2.2:28080/user/password'); // 요청 경로

    try {
      final response = await http.put(
        url,
        headers: {
          'Content-Type': 'application/json',
          'Authorization': 'Bearer $accessToken', // 사용자 access token 포함
        },
        body: jsonEncode({
          'emailToken': emailToken, // 이메일 인증 완료 토큰
          'password': newPassword, // 새로운 비밀번호
        }),
      );

      if (response.statusCode == 200) {
        // 서버 응답이 정상(200)일 경우만
        print('비밀번호 변경 성공: ${response.body}');
      } else {
        print('비밀번호 변경 실패: ${response.statusCode}');
        throw Exception('비밀번호 변경 실패');
      }
    } catch (e) {
      print('네트워크 오류: $e');
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
        backgroundColor: Theme.of(context).scaffoldBackgroundColor,
        body: Center(
          child: Padding(
              padding: const EdgeInsets.symmetric(
                horizontal: 20,
              ),
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                crossAxisAlignment: CrossAxisAlignment.center,
                children: [
                  SizedBox(
                    height: 50,
                    child: introPageHeader(
                      title: '비밀번호 재설정',
                      targetPage: FindPwScreen(),
                    ),
                  ),
                  Text(
                    "8~20자의 영문, 숫자, 특수문자를 포함한 비밀번호를 설정해주세요.",
                    style: TextStyle(
                      color: Color(0xff797979),
                    ),
                  ),
                  SizedBox(height: 30),
                  SizedBox(
                    width: 400,
                    child: Form(
                      key: _formKey,
                      child: Column(
                        children: [
                          TextFormField(
                            controller: _passwordController,
                            obscureText: !_isPasswordVisible,
                            validator: _validatePassword,

                            // build_text_field에 있던 속성 가져옴
                            decoration: InputDecoration(
                              hintText: '새 비밀번호',
                              hintStyle: TextStyle(fontSize: 15),
                              filled: true,
                              fillColor: Colors.white,
                              border: OutlineInputBorder(
                                borderRadius:
                                    BorderRadius.circular(12.0), // 테두리를 둥글게 설정
                              ),
                              enabledBorder: OutlineInputBorder(
                                borderRadius: BorderRadius.circular(12.0),
                                borderSide: BorderSide(
                                    color: Colors.grey.shade400), // 기본 테두리 색
                              ),
                              focusedBorder: OutlineInputBorder(
                                borderRadius: BorderRadius.circular(12.0),
                                borderSide: BorderSide(
                                    color: Color(0xFF424242),
                                    width: 2.0), // 포커스 시 테두리 색
                              ),
                              suffixIcon: Padding(
                                padding: const EdgeInsets.only(right: 8.0),
                                child: IconButton(
                                  icon: Icon(
                                    _isPasswordVisible
                                        ? Icons.visibility_off
                                        : Icons.visibility,
                                  ),
                                  onPressed: () {
                                    setState(
                                      () {
                                        _isPasswordVisible =
                                            !_isPasswordVisible;
                                      },
                                    );
                                  },
                                ),
                              ),
                            ),
                          ),
                          SizedBox(height: 22),
                          TextFormField(
                            controller: _confirmPasswordController,
                            obscureText: !_isConfirmPasswordVisible,
                            validator: _validateConfirmPassword,
                            decoration: InputDecoration(
                              hintText: '새 비밀번호 확인',
                              hintStyle: TextStyle(fontSize: 15),
                              filled: true,
                              fillColor: Colors.white,
                              border: OutlineInputBorder(
                                borderRadius:
                                    BorderRadius.circular(12.0), // 테두리를 둥글게 설정
                              ),
                              enabledBorder: OutlineInputBorder(
                                borderRadius: BorderRadius.circular(12.0),
                                borderSide: BorderSide(
                                    color: Colors.grey.shade400), // 기본 테두리 색
                              ),
                              focusedBorder: OutlineInputBorder(
                                borderRadius: BorderRadius.circular(12.0),
                                borderSide: BorderSide(
                                    color: Color(0xFF424242),
                                    width: 2.0), // 포커스 시 테두리 색
                              ),
                              suffixIcon: Padding(
                                padding: const EdgeInsets.only(right: 8.0),
                                child: IconButton(
                                  icon: Icon(
                                    _isConfirmPasswordVisible
                                        ? Icons.visibility_off
                                        : Icons.visibility,
                                  ),
                                  onPressed: () {
                                    setState(
                                      () {
                                        _isConfirmPasswordVisible =
                                            !_isConfirmPasswordVisible;
                                      },
                                    );
                                  },
                                ),
                              ),
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                  SizedBox(height: 34),
                  SizedBox(
                      width: 400,
                      child: ElevatedButton(
                        onPressed: () {
                          if (_formKey.currentState!.validate()) {
                            // 비밀번호 변경 완료하면 팝업창 띄움, 이후에 홈으로 이동 기능 추가해야 함
                            showDialog(
                              context: context,
                              builder: (BuildContext context) {
                                return AlertDialog(
                                  title: Text('완료'),
                                  content: Text('비밀번호가 성공적으로 변경되었습니다.'),
                                  actions: [
                                    TextButton(
                                      onPressed: () {
                                        Navigator.of(context).pop();
                                      },
                                      child: Text('확인'),
                                    ),
                                  ],
                                );
                              },
                            );
                          }
                        },
                        style: ElevatedButton.styleFrom(
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(12.0),
                          ),
                          padding: EdgeInsets.symmetric(vertical: 25.0),
                          backgroundColor: Color(0xff424242),
                        ),
                        child: Text(
                          '비밀번호 변경',
                          style: TextStyle(
                            fontSize: 15.0,
                            color: Color(0xffffffff),
                          ),
                        ),
                      ))
                ],
              )),
        ));
  }
}
