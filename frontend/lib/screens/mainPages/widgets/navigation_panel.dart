import 'package:flutter/material.dart';
import 'package:font_awesome_flutter/font_awesome_flutter.dart';
import 'package:flutter_screenutil/flutter_screenutil.dart';

class NavigationPanel extends StatelessWidget {
  final int selectedIndex; // 선택된 메뉴 인덱스
  final Function(int) onItemSelected; // 메뉴 선택 시 호출되는 함수

  const NavigationPanel({
    required this.selectedIndex,
    required this.onItemSelected,
  });

  @override
  Widget build(BuildContext context) {
    double screenHeight = MediaQuery.of(context).size.height;
    double bottomPadding = MediaQuery.of(context).padding.bottom;

    // 상단 로고 및 마진
    double headerHeight = 35.h + (screenHeight * 0.11) + 10.h; // 로고 높이를 비율로
    double otherHeights = headerHeight + bottomPadding;

    // 전체 아이템+마진 영역 높이
    double availableHeight = screenHeight - otherHeights;

    // 아이템 개수
    int itemCount = 5;

    // 아이템 및 마진을 균등 분배
    double itemTotalHeight = availableHeight / itemCount;
    double verticalMargin = itemTotalHeight * 0.035; // 마진 비율 (10%)
    double itemHeight = itemTotalHeight - (2 * verticalMargin); // 아이템 높이

    // 마지막 아이템 아래에 여백을 추가 (safe area padding)
    double lastItemBottomMargin = verticalMargin + bottomPadding;

    return SingleChildScrollView(
      physics: const NeverScrollableScrollPhysics(),
      child: Container(
        width: 105.w,
        height: screenHeight + bottomPadding, // SafeArea까지 포함
        decoration: BoxDecoration(
          color: Colors.white,
          boxShadow: [
            BoxShadow(
                color: Colors.black26, blurRadius: 10, offset: Offset(3, 0))
          ],
        ),
        child: Padding(
          padding: EdgeInsets.symmetric(
              vertical: MediaQuery.of(context).size.height * 0.01,
              horizontal: MediaQuery.of(context).size.width * 0.003),
          child: Column(
            children: [
              SizedBox(height: MediaQuery.of(context).padding.top + 30.h),
              Container(
                alignment: Alignment.center,
                child: Image.asset('assets/images/appLogo.png'),
                height: screenHeight * 0.11, // 로고 높이 비율로
              ),
              SizedBox(height: 10.h),
              _navItem(FaIcon(FontAwesomeIcons.drum), "드럼 기초", 0, itemHeight,
                  verticalMargin),
              _navItem(FaIcon(FontAwesomeIcons.handsClapping), "메트로놈", 1,
                  itemHeight, verticalMargin),
              _navItem(FaIcon(FontAwesomeIcons.music), "패턴 및 필인 연습", 2,
                  itemHeight, verticalMargin),
              _navItem(Icon(Icons.queue_music_rounded, size: 13.sp), "악보 연습", 3,
                  itemHeight, verticalMargin),
              _navItem(FaIcon(FontAwesomeIcons.circleUser), "마이페이지", 4,
                  itemHeight, lastItemBottomMargin),
            ],
          ),
        ),
      ),
    );
  }

  Widget _navItem(Widget icon, String title, int index, double itemHeight,
      double verticalMargin) {
    bool isSelected = selectedIndex == index;

    return GestureDetector(
      onTap: () => onItemSelected(index),
      child: Stack(
        children: [
          if (isSelected)
            Positioned(
              left: 3.w,
              top: 5.h,
              bottom: 4.h,
              child: Container(
                width: 2.8.w,
                decoration: BoxDecoration(
                  color: Color.fromARGB(255, 195, 112, 97),
                  borderRadius: BorderRadius.circular(10),
                ),
              ),
            ),
          Container(
            padding: EdgeInsets.symmetric(horizontal: 20.h, vertical: 6.w),
            margin: EdgeInsets.only(
                top: verticalMargin,
                bottom: verticalMargin,
                left: 9.w,
                right: 4.w),
            decoration: BoxDecoration(
              color: isSelected
                  ? Color.fromARGB(255, 249, 231, 227)
                  : Colors.transparent,
              borderRadius: BorderRadius.circular(10),
              boxShadow: isSelected
                  ? [
                      BoxShadow(
                        color: Colors.black26,
                        blurRadius: 5,
                        spreadRadius: 0.5,
                        offset: Offset(0, 4),
                      ),
                    ]
                  : [],
            ),
            child: Row(
              children: [
                SizedBox(width: 2.w),
                IconTheme(
                  data: IconThemeData(
                    color: isSelected ? Color(0XFFD97D6C) : Color(0XFF646464),
                    size: 10.sp,
                  ),
                  child: icon,
                ),
                SizedBox(width: 10.w),
                Expanded(
                  child: Padding(
                    padding: EdgeInsets.symmetric(vertical: 2.5.h),
                    child: Text(title,
                        textAlign: TextAlign.center,
                        style: TextStyle(
                            fontSize: 7.2.sp,
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
