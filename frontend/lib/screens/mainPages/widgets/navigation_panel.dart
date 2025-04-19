import 'package:flutter/material.dart';
import 'package:font_awesome_flutter/font_awesome_flutter.dart';

class NavigationPanel extends StatelessWidget {
  final int selectedIndex; // 선택된 메뉴 인덱스
  final Function(int) onItemSelected; // 메뉴 선택 시 호출되는 함수

  const NavigationPanel({
    required this.selectedIndex,
    required this.onItemSelected,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      width: 300, // 네비게이션 바 고정 크기
      decoration: BoxDecoration(
        color: Colors.white,
        boxShadow: [
          BoxShadow(color: Colors.black26, blurRadius: 10, offset: Offset(3, 0))
        ],
      ),
      child: Column(
        // 네비게이션 바 메뉴
        children: [
          SizedBox(height: 35),
          Text("알려드럼 로고",
              style: TextStyle(
                  fontSize: 17,
                  fontWeight: FontWeight.bold,
                  color: Colors.redAccent)),
          SizedBox(height: 20),
          // 네비게이션 바 메뉴 아이템
          _navItem(FaIcon(FontAwesomeIcons.drum), "드럼 기초", 0),
          _navItem(FaIcon(FontAwesomeIcons.handsClapping), "메트로놈", 1),
          _navItem(FaIcon(FontAwesomeIcons.music), "패턴 및 필인 연습", 2),
          _navItem(FaIcon(FontAwesomeIcons.sliders), "악보 연습", 3),
          _navItem(FaIcon(FontAwesomeIcons.circleUser), "마이페이지", 4),
        ],
      ),
    );
  }

  // 네비게이션 바 메뉴 아이템 위젯 - 아이콘, 타이틀, 인덱스
  Widget _navItem(Widget icon, String title, int index) {
    bool isSelected = selectedIndex == index;

    return GestureDetector(
      onTap: () => onItemSelected(index),
      child: Stack(
        children: [
          // 강조 바 - 선택되었을 때
          if (isSelected)
            Positioned(
              left: 10, // 왼쪽으로 좀 더 빼줌
              top: 4, // 위쪽 정렬 맞춤
              bottom: 2, // 아래쪽 정렬 맞춤
              child: Container(
                width: 6, // 강조 바 두께
                decoration: BoxDecoration(
                  color: Color.fromARGB(255, 195, 112, 97), // 강조 바 색상
                  borderRadius: BorderRadius.circular(10), // 둥글게 처리
                ),
              ),
            ),

          // 네비게이션 버튼 박스
          Container(
            padding: EdgeInsets.symmetric(horizontal: 15, vertical: 15),
            margin: EdgeInsets.only(top: 4, bottom: 4, left: 25, right: 15),
            decoration: BoxDecoration(
              color: isSelected
                  ? Color.fromARGB(255, 249, 231, 227)
                  : Colors.transparent, // 선택된 아이템 배경색
              borderRadius: BorderRadius.circular(10),
              boxShadow: isSelected
                  ? [
                      BoxShadow(
                        color: Colors.black26, // 그림자 색상
                        blurRadius: 5, // 흐림 정도
                        spreadRadius: 0.5, // 퍼짐 정도
                        offset: Offset(0, 5), // 그림자 조정
                      ),
                    ]
                  : [],
            ),
            child: Row(
              // 네비게이션 버튼 아이콘, 타이틀
              children: [
                SizedBox(width: 15),
                IconTheme(
                  data: IconThemeData(
                    color: isSelected ? Color(0XFFD97D6C) : Color(0XFF646464),
                    size: 30,
                  ),
                  child: icon,
                ),
                SizedBox(width: 17),
                Expanded(
                  child: Padding(
                    padding: const EdgeInsets.symmetric(vertical: 2),
                    child: Text(title,
                        textAlign: TextAlign.center,
                        style: TextStyle(
                            fontSize: 20,
                            fontWeight: FontWeight.bold,
                            color: isSelected
                                ? Color(0XFFD97D6C)
                                : Color(0XFF646464))),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
