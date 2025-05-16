import 'package:flutter/material.dart';
import '../../../models/sheet.dart';
import 'package:flutter_screenutil/flutter_screenutil.dart';

class SheetCard extends StatelessWidget {
  final Sheet sheet;

  const SheetCard({super.key, required this.sheet});

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Padding(
          padding: EdgeInsets.only(bottom: 8.h),
          child: AspectRatio(
            aspectRatio: 0.9.h,
            child: Container(
              decoration: BoxDecoration(
                color: sheet.color,
                borderRadius: BorderRadius.circular(12),
                boxShadow: const [
                  BoxShadow(
                    color: Colors.black26,
                    blurRadius: 12,
                    offset: Offset(0, 6),
                  ),
                ],
              ),
              child: Center(
                child: Icon(
                  Icons.music_note,
                  color: Colors.black.withOpacity(0.4),
                  size: 15.sp,
                ),
              ),
            ),
          ),
        ),
        SizedBox(height: 8.h),
        Text(
          sheet.title,
          style: TextStyle(fontSize: 5.sp),
          textAlign: TextAlign.center,
          maxLines: 2,
          overflow: TextOverflow.ellipsis,
        ),
        Text(
          '${sheet.createdDate.year}.${sheet.createdDate.month.toString().padLeft(2, '0')}.${sheet.createdDate.day.toString().padLeft(2, '0')}',
          style: TextStyle(fontSize: 4.5.sp, color: Colors.grey),
          textAlign: TextAlign.center,
        ),
      ],
    );
  }
}
