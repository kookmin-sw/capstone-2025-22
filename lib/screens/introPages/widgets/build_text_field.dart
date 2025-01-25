import 'package:flutter/material.dart';

class buildTextField extends StatelessWidget {
  const buildTextField({
    super.key,
    required this.hint,
    required this.obscureText,
    required this.suffixIcon,
  });

  final String hint;
  final bool obscureText;
  final Widget? suffixIcon;

  @override
  Widget build(BuildContext context) {
    return TextField(
      obscureText: obscureText,
      decoration: InputDecoration(
        hintText: hint,
        hintStyle: TextStyle(fontSize: 15),
        filled: true,
        fillColor: Colors.white, // 배경색
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(12.0), // 테두리를 둥글게 설정
        ),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(12.0),
          borderSide: BorderSide(color: Colors.grey.shade400), // 기본 테두리 색
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(12.0),
          borderSide:
              BorderSide(color: Color(0xFF424242), width: 2.0), // 포커스 시 테두리 색
        ),
        suffixIcon: suffixIcon, // 비밀번호 보기/숨기기 아이콘 추가
      ),
    );
  }
}
