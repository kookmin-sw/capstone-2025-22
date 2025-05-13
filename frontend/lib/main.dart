import 'package:capstone_2025/screens/drumPatternFillPages/practice_result_PP.dart';
import 'package:capstone_2025/screens/drumSheetPages/drum_sheet_player.dart';
import 'package:capstone_2025/screens/drumPatternFillPages/pattern_fill_main.dart';
import 'package:capstone_2025/screens/drumPatternFillPages/pattern_fill_screen.dart';
import 'package:capstone_2025/screens/introPages/login_screen_google.dart';
import 'package:capstone_2025/screens/mainPages/navigation_screens.dart';
import 'package:capstone_2025/screens/drumSheetPages/drum_sheet_player.dart';
import 'package:capstone_2025/services/api_func.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();

  // 앱을 가로 모드로 고정
  SystemChrome.setPreferredOrientations([
    DeviceOrientation.landscapeLeft,
    DeviceOrientation.landscapeRight,
  ]).then((_) {
    runApp(
      ProviderScope(
        child: const MyApp(),
      ),
    );
  });
}

class MyApp extends StatefulWidget {
  const MyApp({super.key});

  @override
  State<MyApp> createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  final _storage = FlutterSecureStorage();
  bool? _isLoggedIn;

  @override
  void initState() {
    super.initState();
    _checkLoginStatus();
  }

  Future<void> _checkLoginStatus() async {
    try {
      final token = await _storage.read(key: 'access_token');
      if (token == null || token.isEmpty) {
        setState(() {
          _isLoggedIn = false;
        });
        return;
      }
      final response =
          await getHTTP('/auth/check', {}, reqHeader: {'authorization': token});
      final isTokenValid = response['body'] == 'valid';
      setState(() {
        _isLoggedIn = token.isNotEmpty && isTokenValid;
      });
    } catch (e) {
      print("자동 로그인 중 예외 발생: $e");
      setState(() {
        _isLoggedIn = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    if (_isLoggedIn == null) {
      // 자동로그인 로직
      return const MaterialApp(
        home: Scaffold(
          body: Center(child: CircularProgressIndicator()),
        ),
      );
    }

    return MaterialApp(
      theme: ThemeData(scaffoldBackgroundColor: Color(0xFFF2F1F3)),
      // home: _isLoggedIn! ? NavigationScreens() : LoginScreenGoogle(),
      home: DrumSheetPlayer(), // 악보연주페이지 확인용
    );
  }
}
