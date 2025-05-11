import 'package:flutter/material.dart';

class Sheet {
  String title;
  String artistName;
  final DateTime createdDate;
  final DateTime lastPracticedDate;
  Color color;
  bool isSelected;

  Sheet({
    required this.title,
    required this.artistName,
    required this.createdDate,
    required this.lastPracticedDate,
    this.color = const Color(0xFFBEBEBE),
    this.isSelected = false,
  });

  factory Sheet.fromJson(Map<String, dynamic> json) {
    return Sheet(
      title: json['sheetName'] ?? "error",
      artistName: json['artistName'] ?? "errorName",
      createdDate: json['createdDate'] != null
          ? DateTime.parse(json['createdDate'])
          : DateTime.now(),
      lastPracticedDate: json['lastPracticeDate'] != null
          ? DateTime.parse(json['lastPracticeDate'])
          : DateTime.now(),
      color: _hexToColor(json['color'] ?? '#BEBEBE'),
    );
  }

  static Color _hexToColor(String hex) {
    hex = hex.replaceFirst('#', '');
    if (hex.length == 6) hex = 'FF$hex';
    return Color(int.parse(hex, radix: 16));
  }

  static String colorToHex(Color color) {
    return '#${color.value.toRadixString(16).substring(2).toUpperCase()}';
  }
}
