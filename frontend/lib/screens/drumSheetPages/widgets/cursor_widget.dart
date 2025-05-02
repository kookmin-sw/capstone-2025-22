import 'package:flutter/material.dart';
import '../../../models/cursor.dart'; // Cursor 데이터 모델 import

// 내부용: 커서 위치 + 스타일을 지정하는 PositionedContainer
class PositionedContainer extends Positioned {
  PositionedContainer({
    super.key,
    required Cursor cursor,
    BoxDecoration? decoration,
    Widget? child,
    double? height,
  }) : super(
          left: cursor.x,
          top: cursor.y + cursor.h - (height ?? cursor.h),
          child: Container(
            height: height ?? cursor.h,
            width: cursor.w,
            decoration: decoration,
            child: child,
          ),
        );
}

// 실제로 사용할 커서 위젯 (색상 + 스타일 적용)
class CursorWidget extends StatelessWidget {
  final Cursor cursor;
  final double? height;
  final double? imageWidth;
  final double? canvasWidth;
  final BoxDecoration decoration;

  const CursorWidget({
    super.key,
    required this.cursor,
    this.height,
    this.imageWidth,
    this.canvasWidth,
    this.decoration = const BoxDecoration(
      color: Color(0xffe6aaa0),
      borderRadius: BorderRadius.all(Radius.circular(4)),
    ),
  });

  @override
  Widget build(BuildContext context) {
    double adjustedX;

    if (cursor.xRatio != null && canvasWidth != null && imageWidth != null) {
      adjustedX = cursor.xRatio! * imageWidth!;
    } else if (cursor.xRatio != null && imageWidth != null) {
      adjustedX = cursor.xRatio! * imageWidth!;
    } else {
      adjustedX = cursor.x;
    }

    return PositionedContainer(
      cursor: cursor.copyWith(x: adjustedX),
      decoration: decoration,
      height: height,
    );
  }
}
