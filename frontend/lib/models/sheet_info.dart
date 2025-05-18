import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/foundation.dart';
import 'package:xml/xml.dart';
import './cursor.dart';

// 악보 종류 (서버 제공/사용자 업로드)
enum SheetType {
  server,
  user,
}

class SheetInfo {
  final String id, title, artist;
  final int bpm;
  final double canvasHeight; // 캔버스 높이 (악보 이미지 기준)
  final List<Cursor> cursorList; // 커서 리스트
  final Uint8List? fullSheetImage; // PNG 포맷으로 인코딩된 바이트 배열
  final String? xmlData; // MusicXML 원본 텍스트
  final List<Uint8List> lineImages; // 줄 단위 잘라낸 이미지 리스트
  // final List<MusicEntry> musicEntries; // 악보에 있는 음표들 (ts 기반)
  final DateTime createdDate; // 생성일

  SheetInfo({
    required this.id,
    required this.title,
    required this.artist,
    required this.bpm,
    required this.canvasHeight,
    required this.cursorList,
    required this.fullSheetImage,
    required this.xmlData,
    required this.lineImages,
    required this.createdDate,
  });

  // JSON -> 객체로 변환
  factory SheetInfo.fromJson(Map<String, dynamic> json) {
    return SheetInfo(
      id: json['id'] ?? '',
      title: json['title'] ?? '제목 없음',
      artist: json['artist'] ?? '아티스트 없음',
      bpm: json['bpm'] ?? 60,
      canvasHeight: (json['canvasHeight'] as num?)?.toDouble() ?? 0.0,
      cursorList: (json['cursorList'] as List<dynamic>?)
              ?.map((e) => Cursor.fromJson(e))
              .toList() ??
          [],
      fullSheetImage:
          json['sheetImage'] != null ? base64Decode(json['sheetImage']) : null,
      xmlData: json['xmlData'] != null
          ? utf8.decode(base64Decode(json['xmlData'] as String))
          : null,
      lineImages: (json['lineImages'] as List<dynamic>?)
              ?.map((e) => base64Decode(e))
              .toList() ??
          [],
      createdDate:
          DateTime.parse(json['createdAt'] ?? DateTime.now().toIso8601String()),
    );
  }

  // 객체 -> JSON으로 변환
  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'title': title,
      'artist': artist,
      'bpm': bpm,
      'canvasHeight': canvasHeight,
      'cursorList': cursorList.map((e) => e.toJson()).toList(),
      'fullSheetImage':
          fullSheetImage != null ? base64Encode(fullSheetImage!) : null,
      'xmlData': xmlData != null ? base64Encode(utf8.encode(xmlData!)) : null,
      'lineImages': lineImages.map((e) => base64Encode(e)).toList(),
      'createdAt': createdDate.toIso8601String(),
    };
  }

  // 복제 (copyWith)
  SheetInfo copyWith({
    String? id,
    String? title,
    String? artist,
    int? bpm,
    double? canvasHeight,
    List<Cursor>? cursorList,
    Uint8List? fullSheetImage,
    List<Uint8List>? lineImages,
    String? xmlData,
    DateTime? createdDate,
  }) {
    return SheetInfo(
        id: id ?? this.id,
        title: title ?? this.title,
        artist: artist ?? this.artist,
        bpm: bpm ?? this.bpm,
        canvasHeight: canvasHeight ?? this.canvasHeight,
        cursorList: cursorList ?? this.cursorList,
        fullSheetImage: fullSheetImage ?? this.fullSheetImage,
        xmlData: xmlData ?? this.xmlData,
        lineImages: lineImages ?? this.lineImages,
        createdDate: createdDate ?? this.createdDate);
  }
}
