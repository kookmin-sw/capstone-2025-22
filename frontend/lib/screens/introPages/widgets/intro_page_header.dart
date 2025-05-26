import 'package:flutter/material.dart';
import 'package:flutter_screenutil/flutter_screenutil.dart';

// intro 페이지 헤더 : 뒤로가기 버튼 + 제목
// 제목과 이동할 페이지를 인자로 받음
class introPageHeader extends StatelessWidget {
  final String title;
  final Widget targetPage; // 이동할 페이지
  final bool previous; // 바로 이전의 페이지로 이동할지 여부

  const introPageHeader({
    super.key,
    required this.title,
    required this.targetPage,
    this.previous = false,
  });

  @override
  Widget build(BuildContext context) {
    return Stack(
      children: [
        Positioned(
          top: 25.h,
          left: 10.w,
          child: IconButton(
            onPressed: previous
                ? () {
                    Navigator.pop(context);
                  }
                : () {
                    Navigator.push(
                      context,
                      MaterialPageRoute(builder: (context) => targetPage),
                    );
                  },
            icon: Icon(
              Icons.arrow_back_ios,
              size: 14.sp,
              color: Color(0xff646464),
            ),
          ),
        ),
        Center(
          child: Padding(
            padding: EdgeInsets.only(
              top: title == "회원가입"
                  ? MediaQuery.of(context).size.height * 0.07
                  : MediaQuery.of(context).size.height * 0.14,
            ),
            child: Text(
              title,
              style: TextStyle(
                fontSize: 11.5.sp,
                fontWeight: FontWeight.w800,
                color: Colors.black.withOpacity(0.8),
              ),
            ),
          ),
        ),
      ],
    );
  }
}
