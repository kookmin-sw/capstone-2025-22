import 'package:flutter/material.dart';

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
    return Row(
      crossAxisAlignment: CrossAxisAlignment.center,
      children: [
        // 뒤로가기 버튼: 화면의 flex 1 차지
        Expanded(
          flex: 1,
          child: Align(
            alignment: Alignment.centerLeft,
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
                Icons.arrow_back_sharp,
                size: 65,
                color: Colors.black,
              ),
            ),
          ),
        ),
        // 제목: 화면의 flex 2 차지
        Expanded(
          flex: 1,
          child: Center(
            child: Text(
              title,
              style: TextStyle(
                fontSize: 30,
                fontWeight: FontWeight.w900,
              ),
            ),
          ),
        ),
        // 오른쪽 빈 공간: 화면의 flex 1 차지
        Expanded(
          flex: 1,
          child: SizedBox(), // 빈 공간
        ),
      ],
    );
  }
}
