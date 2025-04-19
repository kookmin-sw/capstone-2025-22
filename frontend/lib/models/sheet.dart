import 'package:flutter/material.dart';

class Sheet {
  String title;
  final DateTime createdDate;
  final DateTime lastPracticedDate;
  Color color;
  bool isSelected;

  Sheet({
    required this.title,
    required this.createdDate,
    required this.lastPracticedDate,
    this.color = const Color(0xFFBEBEBE),
    this.isSelected = false,
  });

  factory Sheet.fromJson(Map<String, dynamic> json) {
    return Sheet(
      title: json['title'],
      createdDate: DateTime.now(),
      lastPracticedDate: DateTime.parse(json['lastPracticedDate']),
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
