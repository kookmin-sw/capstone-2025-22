import 'package:capstone_2025/screens/introPages/set_new_pw_screen.dart';
import 'package:capstone_2025/screens/mainPages/edit_profile_screen.dart';
import 'package:flutter/material.dart';
import 'package:font_awesome_flutter/font_awesome_flutter.dart';

class MyPage extends StatefulWidget {
  const MyPage({super.key});

  @override
  State<MyPage> createState() => _MyPageState();
}

class _MyPageState extends State<MyPage> {
  List<Map<String, String>> sheetMusicData = [
    {
      "악보명": "그라데이션",
      "마지막 연습 날짜": "2025.01.18",
      "최고 점수": "100",
    },
    {
      "악보명": "Hi Bully",
      "마지막 연습 날짜": "2025.02.10",
      "최고 점수": "95",
    },
    {
      "악보명": "한 페이지가 될 수 있게",
      "마지막 연습 날짜": "2025.02.11",
      "최고 점수": "90",
    },
    {
      "악보명": "개화",
      "마지막 연습 날짜": "2025.02.12",
      "최고 점수": "85",
    },
    {
      "악보명": "곰 세 마리",
      "마지막 연습 날짜": "2025.02.17",
      "최고 점수": "80",
    },
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Color(0xFFF8F4F0),
      body: Center(
        child: Padding(
          padding: EdgeInsets.all(20),
          child: Column(
            children: [
              SizedBox(height: 15),
              _buildSheetMusicHeader(),
              isSheetMusicUploaded
                  ? Expanded(child: _buildSheetMusicTable())
                  : _buildNoSheetMusicMessage(),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildSheetMusicHeader() {
    return Padding(
      padding: const EdgeInsets.only(left: 20),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(Icons.queue_music_rounded, size: 32, color: Color(0xff646464)),
          SizedBox(width: 7),
          Text("악보 연습 기록",
              style: TextStyle(
                  fontSize: 20,
                  fontWeight: FontWeight.bold,
                  color: Color(0xff646464))),
        ],
      ),
    );
  }

  Widget _buildSheetMusicTable() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Container(
          padding: EdgeInsets.symmetric(vertical: 12),
          decoration: BoxDecoration(
            color: Color(0xffD97D6C),
            borderRadius: BorderRadius.vertical(top: Radius.circular(10)),
          ),
          child: Row(
            children: [
              _buildListHeaderCell("악보명", flex: 5),
              _buildListHeaderCell("마지막 연습 날짜", flex: 5),
              _buildListHeaderCell("최고 점수", flex: 3),
            ],
          ),
        ),
        Expanded(
          child: ListView.builder(
            itemCount: sheetMusicData.length,
            padding: EdgeInsets.zero,
            physics: ClampingScrollPhysics(),
            itemBuilder: (context, index) {
              var item = sheetMusicData[index];

              return Container(
                padding: EdgeInsets.symmetric(vertical: 12),
                decoration: BoxDecoration(
                  color: index.isEven ? Colors.white : Colors.grey.shade100,
                  border: Border(
                      bottom:
                          BorderSide(color: Colors.grey.shade300, width: 1)),
                ),
                child: Row(
                  children: [
                    _buildListCell(item["악보명"]!, flex: 5),
                    _buildListCell(item["마지막 연습 날짜"]!, flex: 5),
                    _buildListCell(item["최고 점수"]!, flex: 3),
                  ],
                ),
              );
            },
          ),
        ),
      ],
    );
  }
}

Widget _buildListHeaderCell(String text, {int flex = 1}) {
  return Expanded(
    flex: flex,
    child: Center(
      child: Text(
        text,
        style: TextStyle(fontWeight: FontWeight.bold, color: Colors.white),
      ),
    ),
  );
}

Widget _buildListCell(String text, {int flex = 1}) {
  return Expanded(
    flex: flex,
    child: Center(
      child: Text(
        text,
        style: TextStyle(fontSize: 16),
      ),
    ),
  );
}
