import 'package:flutter/material.dart';
import '../../../models/sheet.dart';

class SheetCard extends StatelessWidget {
  final Sheet sheet;

  const SheetCard({super.key, required this.sheet});

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Expanded(
          child: Container(
            decoration: BoxDecoration(
              color: sheet.color,
              borderRadius: BorderRadius.circular(12),
              boxShadow: const [
                BoxShadow(
                  color: Colors.black26,
                  blurRadius: 12,
                  offset: Offset(0, 6),
                ),
              ],
            ),
            child: Column(
              children: [
                Expanded(
                  child: Center(
                    child: Icon(
                      Icons.music_note,
                      color: Colors.black.withOpacity(0.4),
                      size: 36,
                    ),
                  ),
                ),
              ],
            ),
          ),
        ),
        const SizedBox(height: 8),
        Text(sheet.title, style: const TextStyle(fontSize: 14)),
        Text(
          '${sheet.createdDate.year}.${sheet.createdDate.month.toString().padLeft(2, '0')}.${sheet.createdDate.day.toString().padLeft(2, '0')}',
          style: const TextStyle(fontSize: 12, color: Colors.grey),
        ),
      ],
    );
  }
}
