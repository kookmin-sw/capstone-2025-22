import 'dart:typed_data';
import 'package:flutter/foundation.dart';
import 'package:image/image.dart' as imglib;

// Isolate 에 전달할 파라미터
class CropParams {
  final Uint8List image;
  final int totalLines;
  CropParams(this.image, this.totalLines);
}

// Compute 함수: 이 함수가 실제로 별도 Isolate 에서 실행됩니다
List<Uint8List> cropLinesEntry(CropParams params) {
  final src = imglib.decodeImage(params.image)!;
  final lineHeight = (src.height / params.totalLines).floor();

  List<Uint8List> croppedLines = [];
  for (var i = 0; i < params.totalLines; i++) {
    final cropped = imglib.copyCrop(
      src,
      x: 0,
      y: lineHeight * i + 5,
      width: src.width,
      height: lineHeight,
      antialias: true,
    );
    croppedLines.add(Uint8List.fromList(imglib.encodePng(cropped)));
  }
  return croppedLines;
}

// 동기 버전 일단 테스트용으로 놔둠
List<Uint8List> cropLines(
  Uint8List fullImage, {
  required int totalLines,
  double scale = 1,
}) {
  final src = imglib.decodeImage(fullImage)!;
  final lineHeight = (src.height / totalLines).floor(); // 한 줄 높이

  List<Uint8List> croppedLines = [];
  for (int i = 0; i < totalLines; i++) {
    final cropped = imglib.copyCrop(
      src,
      x: 0,
      y: lineHeight * i + 20,
      width: src.width,
      height: lineHeight,
      antialias: true,
    );
    croppedLines.add(Uint8List.fromList(imglib.encodePng(cropped)));
  }

  return croppedLines;
}
