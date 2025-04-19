import 'package:flutter/material.dart';

class CustomModal extends StatelessWidget {
  final Offset? anchorPoint;
  final List<Widget> children;
  final double? width;
  final double? height;
  final Color? backgroundColor;
  final double borderRadius;
  final EdgeInsets? padding;

  const CustomModal({
    super.key,
    this.anchorPoint,
    required this.children,
    this.width,
    this.height,
    this.backgroundColor,
    this.borderRadius = 15,
    this.padding,
  });

  static void show({
    required BuildContext context,
    required List<Widget> children,
    Offset? anchorPoint,
    double? width,
    double? height,
    Color? backgroundColor,
    double borderRadius = 15,
    EdgeInsets? padding,
  }) {
    showDialog(
      context: context,
      barrierColor: Colors.black54,
      anchorPoint: anchorPoint,
      builder: (BuildContext context) {
        return CustomModal(
          anchorPoint: anchorPoint,
          children: children,
          width: width,
          height: height,
          backgroundColor: backgroundColor,
          borderRadius: borderRadius,
          padding: padding,
        );
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    return Dialog(
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(borderRadius),
      ),
      child: Container(
        width: width ?? 300,
        height: height,
        padding:
            padding ?? const EdgeInsets.symmetric(vertical: 15, horizontal: 20),
        decoration: BoxDecoration(
          color: backgroundColor ?? Colors.white,
          borderRadius: BorderRadius.circular(borderRadius),
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: children,
        ),
      ),
    );
  }
}
