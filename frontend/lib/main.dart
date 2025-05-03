import 'package:capstone_2025/screens/drumBasicsPages/drum_basics_page.dart';
import 'package:capstone_2025/screens/drumPatternFillPages/practice_result_PP.dart';
import 'package:capstone_2025/screens/drumSheetPages/drum_sheet_player.dart';
import 'package:capstone_2025/screens/drumSheetPages/practice_result_MS.dart';
import 'package:capstone_2025/screens/introPages/login_screen_google.dart';
import 'package:capstone_2025/screens/mainPages/navigation_screens.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:capstone_2025/screens/mainPages/navigation_screens.dart';
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
    final token = await _storage.read(key: 'access_token'); // 자동 로그인 여부
    setState(() {
      _isLoggedIn = token != null && token.isNotEmpty;
    });
  }

  @override
  Widget build(BuildContext context) {
    // if (_isLoggedIn == null) { // 자동로그인 로직
    //   return const MaterialApp(
    //     home: Scaffold(
    //       body: Center(child: CircularProgressIndicator()),
    //     ),
    //   );
    // }

    // return MaterialApp(
    //   theme: ThemeData(scaffoldBackgroundColor: Color(0xFFF2F1F3)),
    //   home: _isLoggedIn! ? NavigationScreens() : LoginScreenGoogle(),
    // );
    return MaterialApp(
      theme: ThemeData(scaffoldBackgroundColor: const Color(0xFFF2F1F3)),
      home: const NavigationScreens(),
    );
  }
}
