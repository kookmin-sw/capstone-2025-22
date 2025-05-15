import 'package:capstone_2025/screens/introPages/find_pw_screen.dart';
import 'package:capstone_2025/screens/introPages/login_screen_google.dart';
import 'package:capstone_2025/screens/introPages/set_new_pw_screen.dart';
import 'package:capstone_2025/screens/mainPages/edit_profile_screen.dart';
import 'package:capstone_2025/screens/mainPages/musicsheet_detail.dart';
import 'package:capstone_2025/services/api_func.dart';
import 'package:capstone_2025/services/storage_service.dart';
import 'package:flutter/material.dart';
import 'package:font_awesome_flutter/font_awesome_flutter.dart';
import 'dart:convert'; // Base64 디코딩
import 'dart:typed_data'; // Uint8List 변환

class MyPage extends StatefulWidget {
  const MyPage({super.key});

  @override
  State<MyPage> createState() => _MyPageState();
}

class _MyPageState extends State<MyPage> {
  bool _isLoading = true;
  @override
  void initState() {
    super.initState();
    _loadUserData(); // 페이지가 열릴 때 사용자 데이터 불러오기
  }

  bool isSheetMusicUploaded = true; // 업로드된 악보 존재하는지
  // 사용자 프로필에 출력될 사용자 정보 및 액세스 토큰
  String? email;
  String? userName;
  String? accessToken;
  String? profileImage = null;

  // 악보 기록 아이콘
  FaIcon sheetIcon = FaIcon(
    FontAwesomeIcons.fileLines,
    size: 25,
  );

  List<Map<String, String>> sheetMusicData = []; // 악보 연습 기록 데이터

  // Secure Storage에서 데이터 불러와서 상태 업데이트
  Future<void> _loadUserData() async {
    setState(() => _isLoading = true);
    // Secure Storage에서 사용자 데이터 불러오기
    String? storedEmail = await storage.read(key: 'user_email');
    String? storedUserName = await storage.read(key: 'nick_name');
    String? storedAccessToken = await storage.read(key: 'access_token');
    String? storedProfileImage = await storage.read(key: "profile_image");

    Map<String, String> infoQueryParam = {
      "email": storedEmail ?? "",
    };
    if (infoQueryParam["email"] == "") {
      print("이메일 정보가 없습니다.");
      setState(() => _isLoading = false);
      return;
    }

    if (storedAccessToken == null) {
      print("액세스 토큰 정보가 없습니다.");
      setState(() => _isLoading = false);
      return;
    }

    var clientInfo = await getHTTP("/users/email", infoQueryParam);

    if (clientInfo['errMessage'] == null) {
      // 정상적으로 정보 받아온 경우
      if (!mounted) {
        setState(() => _isLoading = false);
        return;
      }
      setState(() {
        profileImage = clientInfo["body"]["profileImage"];
        email = clientInfo["body"]["email"];
        userName = clientInfo["body"]["nickname"];
      });

      await _createInfoList();
    } else {
      print("프로필 이미지 정보가 없습니다.");
    }
    setState(() => _isLoading = false);
  }

  Future<void> _createInfoList() async {
    final response = await getHTTP(
      '/sheets/practices/representative',
      {'email': email},
    );

    if (response['errMessage'] == null &&
        (response['body'] as List).isNotEmpty) {
      // 악보 연습 기록이 존재하는 경우
      if (!mounted) return;
      setState(() {
        isSheetMusicUploaded = true;
        sheetMusicData = (response['body'] as List)
            .whereType<Map<String, dynamic>>() // null이거나 잘못된 타입 제거
            .map<Map<String, String>>((item) {
          final rawDate =
              DateTime.tryParse(item['lastPracticeDate']?.toString() ?? '');
          final formattedDate = rawDate != null
              ? "${rawDate.year.toString().padLeft(4, '0')}.${rawDate.month.toString().padLeft(2, '0')}.${rawDate.day.toString().padLeft(2, '0')}"
              : '-';

          return {
            "id": item["userSheetId"].toString(),
            "악보명": item["sheetName"] ?? "제목 없음",
            "마지막 연습 날짜": formattedDate,
            "최고 점수": item["maxScore"]?.toString() ?? "-",
          };
        }).toList();
      });
    } else {
      // 악보 연습 기록이 없는 경우
      if (!mounted) return;
      setState(() {
        isSheetMusicUploaded = false;
        sheetMusicData = [];
      });
    }
  }

  // 회원정보 수정 화면으로 이동
  void _navigateToEditProfile() {
    Navigator.push(
      context,
      MaterialPageRoute(builder: (_) => EditProfileScreen()),
    );
  }

  // 비밀번호 변경 화면으로 이동
  void _navigateToChangePassword() {
    Navigator.push(
      context,
      MaterialPageRoute(builder: (_) => FindPwScreen()),
    );
  }

  // 편집 버튼 클릭 시 메뉴 모달 표시
  void _showCustomModal(BuildContext context) {
    showDialog(
      context: context,
      barrierColor: Colors.black54, // 배경 흐림 효과
      builder: (BuildContext context) {
        return Dialog(
          // 모달 다이얼로그
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(15),
          ),
          child: Container(
            // 모달 내용
            width: 300,
            padding: EdgeInsets.symmetric(vertical: 15, horizontal: 20),
            decoration: BoxDecoration(
              color: Colors.white,
              borderRadius: BorderRadius.circular(15),
            ),
            child: Column(
              // 모달 내용
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
      // 클릭 가능 위젯
      onTap: () {
        Navigator.pop(context);
        action();
      },
      child: Padding(
        padding: EdgeInsets.symmetric(vertical: 12),
        child: Row(
          children: [
            Icon(icon, color: const Color(0xFF646464), size: 24),
            SizedBox(width: 10),
            Text(text,
                style: TextStyle(fontSize: 18, fontWeight: FontWeight.w500)),
          ],
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Color(0xFFF8F4F0),
      body: _isLoading
          ? Center(child: CircularProgressIndicator())
          : Center(
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
        boxShadow: [
          BoxShadow(
            color: Colors.grey.withOpacity(0.2),
            spreadRadius: 1,
            blurRadius: 5,
            offset: Offset(0, 3),
          ),
        ],
      ),
      child: Row(
        children: [
          Padding(
            padding: const EdgeInsets.all(8.0),
            child: SizedBox(
              height: 70,
              width: 70,
              child: (profileImage == null)
                  ? CircleAvatar(
                      // 프로필 이미지 - 아이콘 처리(사진으로 바꿔야 함)
                      radius: 40,
                      backgroundColor: Colors.grey[300],
                      child: Icon(Icons.person, size: 60, color: Colors.white),
                    )
                  : CircleAvatar(
                      // 프로필 이미지 - 사진 처리
                      radius: 40,
                      backgroundColor: Colors.grey[300],
                      backgroundImage: MemoryImage(
                          Uint8List.fromList(base64Decode(profileImage!))),
                    ),
            ),
          ),
          SizedBox(width: 20),
          Expanded(
            // 사용자 정보
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Text(userName == null ? "홍길동" : userName!, // 사용자 이름
                        style: TextStyle(
                            fontSize: 23, fontWeight: FontWeight.bold)),
                    SizedBox(width: 10),
                    GestureDetector(
                      // 편집 버튼
                      onTap: () => _showCustomModal(context),
                      child: FaIcon(FontAwesomeIcons.edit,
                          size: 22, color: Colors.black),
                    ),
                  ],
                ),
                Text(email == null ? "example@gmail.com" : email!, // 사용자 이메일
                    style: TextStyle(fontSize: 19, color: Colors.grey)),
              ],
            ),
          ),
          GestureDetector(
            // 로그아웃 버튼
            onTap: () {
              openModal(context);
            },
            child: FaIcon(FontAwesomeIcons.rightFromBracket,
                size: 30, color: Colors.black38),
          ),
          SizedBox(width: 10),
        ],
      ),
    );
  }

  // 악보 연습 기록 타이틀
  Widget _buildSheetMusicHeader() {
    return Padding(
      padding: const EdgeInsets.only(left: 20, bottom: 10),
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
              // 테이블 헤더
              _buildListHeaderCell("악보명", flex: 4),
              _buildListHeaderCell("마지막 연습 날짜", flex: 2),
              _buildListHeaderCell("최고 점수", flex: 2),
              _buildListHeaderCell("상세 기록", flex: 2),
            ],
          ),
        ),
        Expanded(
          child: ListView.builder(
            // 테이블 바디
            itemCount: sheetMusicData.length,
            padding: EdgeInsets.zero,
            physics: ClampingScrollPhysics(), // 스크롤 끝에서 효과 없애기
            itemBuilder: (context, index) {
              // 테이블 행
              var item = sheetMusicData[index];

              return Container(
                // 테이블 셀
                padding: EdgeInsets.symmetric(vertical: 12),
                decoration: BoxDecoration(
                  color: index.isEven ? Colors.white : Colors.grey.shade100,
                  border: Border(
                      bottom:
                          BorderSide(color: Colors.grey.shade300, width: 1)),
                ),
                child: Row(
                  children: [
                    // 테이블 셀 내용
                    _buildListCell(item["악보명"] ?? "", flex: 4),
                    _buildListCell(item["마지막 연습 날짜"] ?? "-", flex: 2),
                    _buildListCell(item["최고 점수"] ?? "-", flex: 2),
                    Expanded(
                      // 상세 기록 버튼
                      flex: 2,
                      child: GestureDetector(
                        onTap: () => {
                          Navigator.push(
                            context,
                            MaterialPageRoute(
                              builder: (context) => MusicsheetDetail(
                                songID: item["id"]!,
                                songTitle: item['악보명'] ?? "-",
                              ),
                            ),
                          ),
                        },
                        child: Center(child: sheetIcon),
                      ),
                    ),
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

// 테이블 헤더 만드는 위젯
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

// 테이블 셀 만드는 위젯
Widget _buildListCell(String text, {int flex = 1}) {
  return Expanded(
    flex: flex,
    child: Center(
      child: Text(
        text,
        textAlign: TextAlign.center,
        style: TextStyle(
          fontSize: 16,
        ),
      ),
    ),
  );
}

Widget modalBtn(BuildContext context, String text, Color backgroundColor,
    bool isTextblack) {
  // 모달 버튼
  return Container(
    width: 155,
    // MediaQuery.of(context).size.width * 0.168,
    height: 50,
    // MediaQuery.of(context).size.height * 0.135,
    alignment: Alignment.center,
    decoration: BoxDecoration(
      color: backgroundColor,
      borderRadius: BorderRadius.circular(15),
    ),
    child: Text(text,
        style: TextStyle(
            color: isTextblack ? Colors.black : Colors.white,
            fontSize: 15,
            fontWeight: FontWeight.w500)),
  );
}

void openModal(
  BuildContext context,
) {
  // 모달 열기
  showDialog(
    context: context,
    builder: (context) => AlertDialog(
      alignment: Alignment.center,
      insetPadding: EdgeInsets.zero,
      contentPadding: EdgeInsets.only(top: 20, bottom: 20),
      backgroundColor: Colors.white,
      content: SizedBox(
        width: 360,
        height: 130,
        child: Column(
          mainAxisAlignment: MainAxisAlignment.spaceAround,
          mainAxisSize: MainAxisSize.min,
          children: [
            SizedBox(
              height: 15,
            ),
            Text(
              '로그아웃 하시겠습니까?',
              textAlign: TextAlign.center,
              style: TextStyle(
                fontSize: 18.5,
                color: Color(0xFF4A4A4A),
                fontWeight: FontWeight.w600,
              ),
            ),
            const SizedBox(height: 20),
            Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                TextButton(
                  style: TextButton.styleFrom(
                    padding: EdgeInsets.zero,
                    minimumSize: Size(0, 0),
                  ),
                  onPressed: () {
                    Navigator.of(context).pop();
                  },
                  child: modalBtn(
                      context, '취소', Color.fromARGB(255, 205, 203, 202), true),
                ),
                SizedBox(width: 10),
                TextButton(
                  style: TextButton.styleFrom(
                    padding: EdgeInsets.zero,
                    minimumSize: Size(0, 0),
                  ),
                  onPressed: () async {
                    String? accessToken =
                        await storage.read(key: 'access_token');

                    // 로그아웃 처리
                    storage.deleteAll();
                    getHTTP('/auth/signout', {}, reqHeader: {
                      'authorization': accessToken ?? "",
                    });
                    Navigator.pushReplacement(
                      context,
                      MaterialPageRoute(
                        builder: (context) => LoginScreenGoogle(),
                      ),
                    );
                  },
                  child: modalBtn(context, '확인', Color(0xffD97D6C), false),
                ),
              ],
            ),
          ],
        ),
      ),
    ),
  );
}
