import 'package:flutter/material.dart';
import 'drum_info_popup.dart';

class DrumBasicsPage extends StatefulWidget {
  const DrumBasicsPage({super.key});

  @override
  State<DrumBasicsPage> createState() => _DrumBasicsPageState();
}

class _DrumBasicsPageState extends State<DrumBasicsPage> {
  final List<Map<String, String>> stepInfo = const [
    {
      'title': 'STEP 1  드럼 종류 알기',
      'subtitle': '드럼 북에는 어떤 종류가 있는지 알아보자!',
      'popupTitle': '드럼 종류',
    },
    {
      'title': 'STEP 2  드럼 용어 알기',
      'subtitle': '드럼 연주를 위해 필요한 용어를 알아보자!',
      'popupTitle': '드럼 용어 정리',
    },
    {
      'title': 'STEP 3  드럼 악보 알기',
      'subtitle': '드럼 악보를 어떻게 읽는지 알아보자!',
      'popupTitle': '드럼 악보 읽기',
    },
    {
      'title': 'STEP 4  드럼 스틱 잡기',
      'subtitle': '드럼 스틱을 잡는 올바른 방법을 익혀보자!',
      'popupTitle': '드럼 스틱 잡는 방법',
    },
  ];

  void _showPopup(String title) {
    showDialog(
      context: context,
      builder: (_) => DrumInfoPopup(
        title: title,
        imagePath: null,
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 32.0, vertical: 40.0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // 상단 타이틀
          Center(
            child: Text(
              '드럼 기초',
              style: TextStyle(
                  fontSize: 26,
                  color: Color(0xff595959),
                  fontWeight: FontWeight.w800),
            ),
          ),

          // 카드 리스트
          Expanded(
            child: ListView.builder(
              itemCount: stepInfo.length,
              itemBuilder: (context, index) {
                final step = stepInfo[index];
                final bool isLast = index == stepInfo.length - 1;
                return Padding(
                  padding: EdgeInsets.only(
                    bottom: isLast ? 70.0 : 13.0,
                  ),
                  child: Container(
                    decoration: BoxDecoration(
                      color: Colors.white,
                      borderRadius: BorderRadius.circular(12.0),
                      boxShadow: [
                        BoxShadow(
                          color: Colors.grey.withOpacity(0.2),
                          blurRadius: 6,
                          offset: Offset(0, 3),
                        )
                      ],
                    ),
                    child: Row(
                      children: [
                        // 왼쪽 포인트 바
                        Container(
                          width: 12,
                          height: 80,
                          decoration: const BoxDecoration(
                            color: Color(0xFFD97D6C),
                            borderRadius: BorderRadius.only(
                              topLeft: Radius.circular(10),
                              bottomLeft: Radius.circular(10),
                            ),
                          ),
                        ),

                        // 텍스트 내용
                        Expanded(
                          child: Padding(
                            padding: const EdgeInsets.only(
                              left: 20.0,
                              top: 17,
                              bottom: 18,
                            ),
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Text(
                                  step['title']!,
                                  style: const TextStyle(
                                    fontWeight: FontWeight.w800,
                                    fontSize: 16,
                                    color: Color(0xFF595959),
                                  ),
                                ),
                                Text(
                                  step['subtitle']!,
                                  style: const TextStyle(
                                    fontSize: 14,
                                    color: Color(0xFF949494),
                                  ),
                                ),
                              ],
                            ),
                          ),
                        ),

                        // 오른쪽 > 버튼
                        IconButton(
                          icon: const Icon(Icons.chevron_right),
                          onPressed: () => _showPopup(step['popupTitle']!),
                          color: Colors.black45,
                          iconSize: 26,
                          padding: const EdgeInsets.only(right: 20),
                        ),
                      ],
                    ),
                  ),
                );
              },
            ),
          ),
        ],
      ),
    );
  }
}
