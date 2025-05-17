// lib/screens/introPages/splash_screen.dart
import 'dart:async';
import 'package:flutter/material.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import 'package:capstone_2025/services/api_func.dart';
import 'package:capstone_2025/screens/introPages/login_screen.dart';
import 'package:capstone_2025/screens/mainPages/navigation_screens.dart';

class SplashScreen extends StatefulWidget {
  const SplashScreen({Key? key}) : super(key: key);
  @override
  State<SplashScreen> createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen> {
  final _storage = const FlutterSecureStorage();

  @override
  void initState() {
    super.initState();
    _startUp();
  }

  Future<void> _startUp() async {
    // (1) 최소 0.9초 기다리기
    final delay = Future.delayed(const Duration(milliseconds: 900));
    // (2) 로그인 토큰 읽고 유효성 검사
    final loginCheck = _storage.read(key: 'access_token').then((token) async {
      if (token == null || token.isEmpty) return false;
      final resp = await getHTTP(
        '/auth/check',
        {},
        reqHeader: {'authorization': token},
      );
      return resp['body'] == 'valid';
    }).catchError((_) => false);

    // 두 작업이 다 끝날 때까지 기다리고
    final results = await Future.wait([delay, loginCheck]);
    final isLoggedIn = results[1] as bool;

    // (3) 이제 로그인 또는 메인으로 이동
    Navigator.of(context).pushReplacement(
      MaterialPageRoute(
        builder: (_) =>
            isLoggedIn ? const NavigationScreens() : const LoginScreen(),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFD97D6C),
      body: Center(
        child: Image.asset(
          'assets/images/intro_logo.png',
          width: 100,
        ),
      ),
    );
  }
}
