import 'package:flutter/material.dart';
import 'package:flutter_screenutil/flutter_screenutil.dart';

// ignore: camel_case_types
class buildTextField extends StatelessWidget {
  const buildTextField({
    super.key,
    required this.controller, // 입력 값 관리하는 컨트롤러(ex: _emailController, _passController)
    required this.hint, // 입력 필트 내부에 표시되는 힌트 텍스트
    required this.obscureText, // 비밀번호 입력 시 가릴지 여부
    required this.suffixIcon, // 입력 필드 오른쪽에 표시할 아이콘 (ex: 비밀번호 눈 아이콘)
  });

  final TextEditingController controller;
  final String hint;
  final bool obscureText;
  final Widget? suffixIcon;

  @override
  Widget build(BuildContext context) {
    return TextField(
      controller: controller, // 입력한 값을 가져오는 컨트롤러
      obscureText: obscureText, // true이면 값을 .로 표시(비밀번호 필드)
      decoration: InputDecoration(
        hintText: hint,
        hintStyle: TextStyle(fontSize: 5.5.sp),
        filled: true,
        fillColor: Colors.white, // 배경색 흰색으로 설정
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(12.0), // 테두리를 둥글게 설정
        ),
        enabledBorder: OutlineInputBorder(
          // 입력 필드가 선택되지 않았을 때 표시되는 테두리 스타일
          borderRadius: BorderRadius.circular(12.0),
          borderSide: BorderSide(color: Colors.grey.shade400), // 기본 테두리 색
        ),
        focusedBorder: OutlineInputBorder(
          // 입력 필드가 focus 받았을 때 표시되는 테두리 스타일
          borderRadius: BorderRadius.circular(12.0),
          borderSide:
              BorderSide(color: Color(0xFF424242), width: 2.0), // 포커스 시 테두리 색
        ),
        suffixIcon: suffixIcon, // 비밀번호 보기 or 숨기기 아이콘 추가
      ),
    );
  }
}
