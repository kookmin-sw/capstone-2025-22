import 'package:flutter/material.dart';
import '../../../models/sheet.dart';
import 'package:flutter_screenutil/flutter_screenutil.dart';

class SheetCard extends StatelessWidget {
  final Sheet sheet;
  final Color iconColor;
  final DateTime? displayDate;

  const SheetCard({
    super.key,
    required this.sheet,
    required this.iconColor,
    this.displayDate,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Padding(
          padding: EdgeInsets.only(bottom: 8.h),
          child: AspectRatio(
            aspectRatio: 0.9.h,
            child: Stack(children: [
              Positioned.fill(
                child: Container(
                  decoration: BoxDecoration(
                    color: Colors.white,
                    borderRadius: BorderRadius.circular(12),
                    boxShadow: const [
                      BoxShadow(
                        color: Color(0xFFd9d9d9),
                        blurRadius: 4,
                        offset: Offset(0, 4),
                      ),
                    ],
                  ),
                ),
              ),
              Positioned(
                top: 0,
                left: 0,
                bottom: 0,
                right: 2.5.w,
                child: Container(
                  decoration: BoxDecoration(
                    color: sheet.color,
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Center(
                    child: Icon(
                      Icons.music_note,
                      color: iconColor,
                      size: 15.sp,
                    ),
                  ),
                ),
              ),
            ]),
          ),
        ),
        SizedBox(height: 8.h),
        Tooltip(
          message: sheet.title, // 롱프레스/호버 시 전체 텍스트 표시
          child: Text(
            sheet.title,
            style: TextStyle(
              fontSize: 6.sp,
              fontWeight: FontWeight.w800,
              color: const Color(0xFF646464),
            ),
            textAlign: TextAlign.center,
            maxLines: 1, // 한 줄만 보여주고
            overflow: TextOverflow.ellipsis, // 뒤에 … 처리
          ),
        ),
        Text(
          displayDate != null
              ? '${displayDate!.year}.${displayDate!.month.toString().padLeft(2, '0')}.${displayDate!.day.toString().padLeft(2, '0')}'
              : '',
          style: TextStyle(fontSize: 4.5.sp, color: const Color(0xffA2A2A2)),
          textAlign: TextAlign.center,
        ),
      ],
    );
  }
}
