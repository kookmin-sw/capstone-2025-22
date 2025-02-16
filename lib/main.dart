import 'package:capstone_2025/screens/introPages/login_screen_google.dart';
import 'package:flutter/material.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      // 페이지 공통 백그라운드 컬러 지정
      theme: ThemeData(scaffoldBackgroundColor: Color(0xFFF2F1F3)),
      home: LoginScreenGoogle(),
    );
  }
}
