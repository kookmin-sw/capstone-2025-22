import 'package:flutter/material.dart';
import 'package:flutter_screenutil/flutter_screenutil.dart';

class CompleteDialog extends StatelessWidget {
  const CompleteDialog(
      {super.key,
      required this.mainText,
      required this.subText,
      required this.onClose});

  final String mainText;
  final String subText;
  final VoidCallback onClose;

  @override
  Widget build(BuildContext context) {
    return Dialog(
      backgroundColor: Colors.white,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
      child: Container(
        width: 110.w,
        height: 170.h,
        child: Padding(
          padding: EdgeInsets.symmetric(horizontal: 20.h, vertical: 30.h),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  Text(
                    mainText,
                    textAlign: TextAlign.start,
                    style: TextStyle(
                      fontSize: 6.sp,
                      fontWeight: FontWeight.w800,
                      color: Color(0xff646464),
                    ),
                  ),
                  SizedBox(height: 5.h),
                  Text(
                    subText,
                    textAlign: TextAlign.start,
                    style: TextStyle(
                      fontSize: 5.sp,
                      fontWeight: FontWeight.w300,
                      color: Color(0xff646464),
                    ),
                  ),
                ],
              ),
              SizedBox(height: 20.h),
              Align(
                alignment: Alignment.centerRight,
                child: InkWell(
                  onTap: onClose,
                  child: Text(
                    "확인",
                    style: TextStyle(
                      color: Color(0xFFd97d6c),
                      fontWeight: FontWeight.w600,
                      fontSize: 6.sp,
                    ),
                  ),
                ),
              )
            ],
          ),
        ),
      ),
    );
  }
}
