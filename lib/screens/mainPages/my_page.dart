import 'package:capstone_2025/screens/introPages/set_new_pw_screen.dart';
import 'package:flutter/material.dart';
import 'package:font_awesome_flutter/font_awesome_flutter.dart';

class MyPage extends StatefulWidget {
  const MyPage({super.key});

  @override
  State<MyPage> createState() => _MyPageState();
}

class _MyPageState extends State<MyPage> {
  bool isSheetMusicUploaded = true;

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

  // 편집 버튼 클릭 시 메뉴 모달 표시
  void _showCustomModal(BuildContext context) {
    showDialog(
      context: context,
      barrierColor: Colors.black54, // 배경 흐림 효과
      builder: (BuildContext context) {
        return Dialog(
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(15),
          ),
          child: Container(
            width: 300,
            padding: EdgeInsets.symmetric(vertical: 15, horizontal: 20),
            decoration: BoxDecoration(
              color: Colors.white,
              borderRadius: BorderRadius.circular(15),
            ),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                _buildMenuItem(
                    context, "회원정보 수정", Icons.person, _navigateToEditProfile),
                Divider(),
                _buildMenuItem(
                    context, "비밀번호 변경", Icons.lock, _navigateToChangePassword),
              ],
            ),
          ),
        );
      },
    );
  }

  // 모달 메뉴 아이템
  Widget _buildMenuItem(
      BuildContext context, String text, IconData icon, Function action) {
    return InkWell(
      onTap: () {
        Navigator.pop(context);
        action();
      },
      child: Padding(
        padding: EdgeInsets.symmetric(vertical: 12),
        child: Row(
          children: [
            Icon(icon, color: Colors.black54, size: 24),
            SizedBox(width: 10),
            Text(text,
                style: TextStyle(fontSize: 18, fontWeight: FontWeight.w500)),
          ],
        ),
      ),
    );
  }

  // 회원정보 수정 화면으로 이동
  void _navigateToEditProfile() {
    Navigator.push(
      context,
      MaterialPageRoute(
          builder: (_) => SetNewPwScreen()), // TODO: EditProfileScreen으로 변경 가능
    );
  }

  // 비밀번호 변경 화면으로 이동
  void _navigateToChangePassword() {
    Navigator.push(
      context,
      MaterialPageRoute(builder: (_) => SetNewPwScreen()),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Color(0xFFF8F4F0),
      body: Center(
        child: Padding(
          padding: EdgeInsets.all(20),
          child: Column(
            children: [
              _buildProfileSection(),
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

  // 프로필 섹션
  Widget _buildProfileSection() {
    return Container(
      padding: EdgeInsets.symmetric(vertical: 10, horizontal: 20),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(10),
      ),
      child: Row(
        children: [
          Padding(
            padding: const EdgeInsets.all(8.0),
            child: SizedBox(
              height: 70,
              width: 70,
              child: CircleAvatar(
                radius: 40,
                backgroundColor: Colors.grey[300],
                child: Icon(Icons.person, size: 60, color: Colors.white),
              ),
            ),
          ),
          SizedBox(width: 20),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Text("홍길동",
                        style: TextStyle(
                            fontSize: 23, fontWeight: FontWeight.bold)),
                    SizedBox(width: 10),
                    GestureDetector(
                      onTap: () => _showCustomModal(context),
                      child: FaIcon(FontAwesomeIcons.edit,
                          size: 22, color: Colors.black),
                    ),
                  ],
                ),
                Text("example@gmail.com",
                    style: TextStyle(fontSize: 19, color: Colors.grey)),
              ],
            ),
          ),
        ],
      ),
    );
  }

  // 악보 연습 기록 헤더
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

  // 악보가 없을 때 표시할 메시지
  Widget _buildNoSheetMusicMessage() {
    return Expanded(
      flex: 3,
      child: Padding(
        padding: const EdgeInsets.only(top: 80),
        child: Text(
          "지정된 악보가 없습니다. \n악보 연습에서 악보를 추가해보세요!",
          textAlign: TextAlign.center,
          style: TextStyle(
              fontSize: 22,
              color: Colors.grey.withOpacity(0.7),
              fontWeight: FontWeight.bold),
        ),
      ),
    );
  }

  // 악보 연습 기록 테이블
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
