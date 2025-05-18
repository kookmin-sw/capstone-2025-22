import 'package:flutter/material.dart';

class InnerShadow extends StatelessWidget {
  final Color shadowColor;
  final double blur;
  final Offset offset;
  final BorderRadius borderRadius;
  final Widget child;

  const InnerShadow({
    super.key,
    required this.shadowColor,
    required this.blur,
    required this.offset,
    required this.borderRadius,
    required this.child,
  });

  @override
  Widget build(BuildContext context) {
    return ClipRRect(
      borderRadius: borderRadius,
      child: Stack(
        children: [
          child,
          Positioned.fill(
            child: CustomPaint(
              painter: _InnerShadowPainter(
                shadowColor: shadowColor,
                blur: blur,
                offset: offset,
                borderRadius: borderRadius,
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _InnerShadowPainter extends CustomPainter {
  final Color shadowColor;
  final double blur;
  final Offset offset;
  final BorderRadius borderRadius;

  _InnerShadowPainter({
    required this.shadowColor,
    required this.blur,
    required this.offset,
    required this.borderRadius,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final rect = Offset.zero & size;
    final rrect = borderRadius.toRRect(rect);

    final paint = Paint()
      ..color = shadowColor
      ..maskFilter = MaskFilter.blur(BlurStyle.normal, blur);

    canvas.saveLayer(rect, Paint());
    canvas.translate(offset.dx, offset.dy);
    canvas.drawRRect(rrect, paint);
    canvas.translate(-offset.dx, -offset.dy);

    final clearPaint = Paint()..blendMode = BlendMode.clear;
    canvas.drawRRect(rrect.deflate(blur), clearPaint);
    canvas.restore();
  }

  @override
  bool shouldRepaint(CustomPainter oldDelegate) => false;
}
