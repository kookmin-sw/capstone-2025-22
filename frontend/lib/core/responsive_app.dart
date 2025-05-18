import 'package:flutter/material.dart';
import 'responsive_scaffold.dart';
import 'package:capstone_2025/screens/introPages/login_screen_google.dart';

class ResponsiveApp extends StatelessWidget {
  const ResponsiveApp({super.key});

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, constraints) {
        return OrientationBuilder(
          builder: (context, orientation) {
            return ResponsiveScaffold(
              maxWidth: constraints.maxWidth,
              maxHeight: constraints.maxHeight,
              orientation: orientation,
              body: MaterialApp(
                // main.dart에 있던 theme 설정
                theme:
                    ThemeData(scaffoldBackgroundColor: const Color(0xFFF2F1F3)),
                // 일단 로그인 화면을 home에 지정
                home: const LoginScreenGoogle(),
              ),
            );
          },
        );
      },
    );
  }
}
