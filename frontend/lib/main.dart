import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:capstone_2025/services/api_func.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_screenutil/flutter_screenutil.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import 'package:capstone_2025/screens/introPages/login_screen.dart';
import 'package:capstone_2025/screens/mainPages/navigation_screens.dart';
import 'package:capstone_2025/screens/drumPatternFillPages/pattern_fill_screen.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // 1) Orientation 고정
  await SystemChrome.setPreferredOrientations([
    DeviceOrientation.landscapeLeft,
    DeviceOrientation.landscapeRight,
  ]);

  // 2) Status Bar 숨기기 (및 네비게이션 바도 숨기고 싶으면 overlays: [])
  SystemChrome.setEnabledSystemUIMode(
    SystemUiMode.manual,
    // overlays: [SystemUiOverlay.bottom], // bottom만 두면 네비게이션바만 남고 상태표시줄은 숨김
    overlays: [], // 완전 풀스크린 (네비바까지 모두 숨기고 싶다면
  );

  runApp(
    ProviderScope(child: const MyApp()),
  );
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

    return ScreenUtilInit(
      designSize: Size(375, 812),
      minTextAdapt: true,
      splitScreenMode: true,
      builder: (context, child) {
        return MaterialApp(
          theme: ThemeData(scaffoldBackgroundColor: Color(0xFFF2F1F3)),
          // home: _isLoggedIn! ? NavigationScreens() : LoginScreen(),
          home: PatternFillScreen(index: 1), // 패턴및필인페이지 확인용
        );
      },
    );
  }
}
