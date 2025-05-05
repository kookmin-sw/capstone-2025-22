// BuildContext에 screenW, screenH 확장 메서드 추가하여 화면 크기 조회 간소화
import 'package:flutter/widgets.dart';

extension SizeExt on BuildContext {
  double get screenW => MediaQuery.of(this).size.width;
  double get screenH => MediaQuery.of(this).size.height;
}
