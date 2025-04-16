import 'package:capstone_2025/screens/drumBasicsPages/drum_basics_page.dart';
import 'package:capstone_2025/screens/drumPatternFillPages/pattern_fill_main.dart';
import 'package:capstone_2025/screens/drumSheetPages/drum_sheet_screen.dart';
import 'package:capstone_2025/screens/drumPatternFillPages/test.dart';
import 'package:capstone_2025/screens/mainPages/my_page.dart';
import 'package:capstone_2025/screens/mainPages/widgets/navigation_panel.dart';
import 'package:flutter/material.dart';

class NavigationScreens extends StatefulWidget {
  const NavigationScreens({super.key});

  @override
  _NavigationScreensState createState() => _NavigationScreensState();
}

class _NavigationScreensState extends State<NavigationScreens> {
  int _selectedIndex = 4; // 선택된 메뉴 인덱스 - default: 4

  // 선택된 인덱스에 따라 오른쪽 화면을 변경하는 함수 - 추후 변경 필요
  Widget _getPage(int index) {
    switch (index) {
      case 0: // 드럼 기초
        return DrumBasicsPage();
      case 1: // 메트로놈
        return MyPage();
      case 2: // 패턴 및 필인 연습
        return PatternFillMain(); // 테스트 페이지. 수정 필요
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
            selectedIndex: _selectedIndex, // 선택된 메뉴 인덱스
            onItemSelected: (index) {
              setState(() {
                // 메뉴 선택 시 해당 메뉴 인덱스로 변경
                _selectedIndex = index;
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
              child: _getPage(_selectedIndex),
            ),
          ),
        ],
      ),
    );
  }
}
