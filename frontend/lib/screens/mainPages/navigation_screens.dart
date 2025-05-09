import 'dart:ffi';

import 'package:capstone_2025/screens/drumBasicsPages/drum_basics_page.dart';
import 'package:capstone_2025/screens/drumPatternFillPages/pattern_fill_main.dart';
import 'package:capstone_2025/screens/drumSheetPages/drum_sheet_screen.dart';
import 'package:capstone_2025/screens/mainPages/my_page.dart';
import 'package:capstone_2025/screens/mainPages/widgets/navigation_panel.dart';
import 'package:capstone_2025/screens/metronomePages/metronome.dart';
import 'package:flutter/material.dart';

class NavigationScreens extends StatefulWidget {
  final int firstSelectedIndex; // 선택된 메뉴 인덱스

  const NavigationScreens({super.key, this.firstSelectedIndex = 4});

  @override
  NavigationScreensState createState() => NavigationScreensState();
}

class NavigationScreensState extends State<NavigationScreens> {
  int selectedIndex = 4; // 선택된 메뉴 인덱스 - default: 4

  // 선택된 인덱스에 따라 오른쪽 화면을 변경하는 함수
  Widget _getPage(int index) {
    switch (index) {
      case 0: // 드럼 기초
        return DrumBasicsPage();
      case 1: // 메트로놈
        return Metronome();
      case 2: // 패턴 및 필인 연습
        return PatternFillMain();
      case 3: // 악보 연습
        return DrumSheetScreen();
      case 4: // 마이페이지
        return MyPage();
      default:
        return Center(child: Text("페이지를 선택해주세요"));
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Row(
        children: [
          // 좌측 네비게이션 바 (고정)
          NavigationPanel(
            selectedIndex: selectedIndex, // 선택된 메뉴 인덱스
            onItemSelected: (index) {
              setState(() {
                // 메뉴 선택 시 해당 메뉴 인덱스로 변경
                selectedIndex = index;
              });
            },
          ),

          // 오른쪽 화면 (선택된 메뉴에 따라 변경)
          Expanded(
            child: AnimatedSwitcher(
              // 화면 전환 애니메이션
              duration: Duration(milliseconds: 300), // 부드러운 화면 전환 애니메이션
              transitionBuilder: (Widget child, Animation<double> animation) {
                return FadeTransition(
                    opacity: animation, child: child); // 페이드 효과
              },
              child: _getPage(selectedIndex),
            ),
          ),
        ],
      ),
    );
  }
}
