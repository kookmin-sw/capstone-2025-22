import 'package:flutter/material.dart';

void openMusicSheet(BuildContext context) {
  // 악보 확대 버튼 클릭 시
  showDialog(
    context: context,
    builder: (context) => Dialog(
      insetPadding: EdgeInsets.symmetric(horizontal: 30, vertical: 0),
      backgroundColor: Colors.white,
      child: LayoutBuilder(
        builder: (context, constraints) {
          return SizedBox(
            width: constraints.maxWidth,
            height: constraints.maxHeight,
            child: Stack(
              children: [
                Padding(
                  padding: const EdgeInsets.all(7),
                  child: SingleChildScrollView(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.stretch,
                      children: [
                        Image.asset(
                          'assets/images/image.png',
                          fit: BoxFit.fitWidth,
                        ),
                        Image.asset(
                          'assets/images/image.png',
                          fit: BoxFit.fitWidth,
                        ),
                        Image.asset(
                          'assets/images/image.png',
                          fit: BoxFit.fitWidth,
                        ),
                        // 필요한 만큼 추가 가능
                      ],
                    ),
                  ),
                ),
                IconButton(
                  onPressed: () => {Navigator.pop(context)},
                  icon: Icon(
                    Icons.close_rounded,
                    size: 45,
                    color: Colors.black54,
                  ),
                ),
              ],
            ),
          );
        },
      ),
    ),
  );
}
