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
          top: cursor.y,
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
  final BoxDecoration decoration;

  const CursorWidget({
    super.key,
    required this.cursor,
    this.height,
    this.imageWidth,
    this.decoration = const BoxDecoration(
      color: Color(0xffe6aaa0),
      borderRadius: BorderRadius.all(Radius.circular(4)),
    ),
  });

  @override
  Widget build(BuildContext context) {
    // 1) X 위치 계산: ratio 기반으로 canvasWidth 쓰기
    final double adjustedX = (cursor.xRatio != null && imageWidth != null)
        ? cursor.xRatio! * imageWidth!
        : cursor.x;

    // 2) Y 위치 계산: yRatio가 있으면 height(=imageHeight)*yRatio, 없으면 cursor.y
    final double adjustedY = (cursor.yRatio != null && height != null)
        ? cursor.yRatio! * height!
        : cursor.y;

    // 3) adjustedCursor 로 치환
    final adjustedCursor = cursor.copyWith(
      x: adjustedX,
      y: adjustedY,
    );

    return PositionedContainer(
      cursor: adjustedCursor,
      decoration: decoration,
      height: height,
    );
  }
}
