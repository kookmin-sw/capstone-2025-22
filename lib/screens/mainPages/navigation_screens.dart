import 'package:capstone_2025/screens/drumBasicsPages/drum_basics_page.dart';
import 'package:capstone_2025/screens/mainPages/my_page.dart';
import 'package:capstone_2025/screens/mainPages/widgets/navigation_panel.dart';
import 'package:flutter/material.dart';

class NavigationScreens extends StatefulWidget {
  @override
  _NavigationScreensState createState() => _NavigationScreensState();
}

class _NavigationScreensState extends State<NavigationScreens> {
  int _selectedIndex = 0; // 선택된 메뉴 인덱스

  // 선택된 인덱스에 따라 오른쪽 화면을 변경하는 함수 - 추후 변경 필요
  Widget _getPage(int index) {
    switch (index) {
      case 0:
        return DrumBasicsPage();
      case 1:
        return MyPage();
      case 2:
        return MyPage();
      case 3:
        return MyPage();
      case 4:
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
            selectedIndex: _selectedIndex,
            onItemSelected: (index) {
              setState(() {
                _selectedIndex = index;
              });
            },
          ),

          // 오른쪽 화면 (선택된 메뉴에 따라 변경)
          Expanded(
            child: AnimatedSwitcher(
              duration: Duration(milliseconds: 300), // 부드러운 화면 전환 애니메이션
              transitionBuilder: (Widget child, Animation<double> animation) {
                return FadeTransition(opacity: animation, child: child);
              },
              child: _getPage(_selectedIndex),
            ),
          ),
        ],
      ),
    );
  }
}
