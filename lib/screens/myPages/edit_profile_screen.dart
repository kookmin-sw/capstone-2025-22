import 'dart:io';
import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
// ignore: depend_on_referenced_packages
import 'package:image_picker/image_picker.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import 'package:capstone_2025/screens/introPages/find_pw_screen.dart';
import 'package:capstone_2025/screens/introPages/widgets/intro_page_header.dart';

class EditProfileScreen extends StatefulWidget {
  const EditProfileScreen({super.key});

  @override
  State<EditProfileScreen> createState() => _EditProfileScreenState();
}

class _EditProfileScreenState extends State<EditProfileScreen> {
  final TextEditingController _emailController = TextEditingController();
  final TextEditingController _nicknameController = TextEditingController();
  final _storage = const FlutterSecureStorage(); // Secure Storage 인스턴스

  bool _isDuplicate = false; // 닉네임 중복 여부
  File? _profileImage; // 프로필 사진
  bool _isModified = false; // 회원정보 수정 여부

  // secure storage에서 불러온 데이터 저장
  String? email;
  String? userName;
  String? accessToken;

  @override
  void initState() {
    super.initState();
    _loadUserData(); // secure storage에서 유저 데이터 불러오기

    // 닉네임 변경 시 _checkModifincation 상태 갱신
    _nicknameController.addListener(_checkModifincation);
  }

// Secure Storage에서 데이터 불러와서 입력 필드 초기화
  Future<void> _loadUserData() async {
    email = await _storage.read(key: 'user_email');
    userName = await _storage.read(key: 'user_name');
    accessToken = await _storage.read(key: 'access_token');

    setState(() {
      _emailController.text = email ?? "example@gmail.com";
      _nicknameController.text = userName ?? "홍길동";
    });

    // // 테스트 코드
    // setState(() {
    //   _emailController.text = "bomi0320@naver.com" ?? "example@gmail.com";
    //   _nicknameController.text = "bomi" ?? "홍길동";
    // });
  }

  void _checkModifincation() {
    setState(() {
      _isModified = (_emailController.text != "example@gmail.com" ||
          _nicknameController.text != "홍길동" ||
          _profileImage != null);
    });
  }

// 서버에서 닉네임 중복 확인
  Future<void> _checkNickname() async {
    print(_nicknameController.text);
    final uri = Uri.parse(
        'http://10.0.2.2:28080/verification/nicknames=${_nicknameController.text}');
    final response = await http.get(uri);
    final data = jsonDecode(response.body);
    print(response.statusCode);
    print(data['body']);
    if (response.statusCode == 200) {
      _isDuplicate = true; // 200이면 이미 존재하는 이메일
    }
  }

  // 프로필 사진 선택
  Future<void> _pickImage(ImageSource source) async {
    final picker = ImagePicker();
    final pickedFile = await picker.pickImage(source: source);
    if (pickedFile != null) {
      setState(() {
        _profileImage = File(pickedFile.path);
        _isModified = true; // 정보가 변경됨
      });
    }
  }

  // 프로필 사진 변경 메뉴
  void _showImagePicker(BuildContext context, TapDownDetails details) {
    final RenderBox renderBox = context.findRenderObject() as RenderBox;
    final Offset offset = renderBox.localToGlobal(details.globalPosition);

    showMenu(
      context: context,
      position: RelativeRect.fromLTRB(
        offset.dx, // X 좌표
        offset.dy + 20, // Y 좌표 (조금 더 아래에 위치하도록 조정)
        offset.dx + 500, // 메뉴 너비
        offset.dy + 100, // 메뉴 높이
      ),
      items: [
        PopupMenuItem(
          child: ListTile(
            // leading: Icon(Icons.add_a_photo_outlined),
            title: Text("사진 촬영"),
            onTap: () {
              Navigator.pop(context); // 메뉴 닫기
              _pickImage(ImageSource.camera);
            },
          ),
        ),
        PopupMenuItem(
          child: ListTile(
            // leading: Icon(Icons.photo),
            title: Text("앨범에서 선택"),
            onTap: () {
              Navigator.pop(context); // 메뉴 닫기
              _pickImage(ImageSource.gallery);
            },
          ),
        ),
      ],
    );
  }

  Future<void> changeUserProfile() async {
    final url = Uri.parse('http://10.0.2.2:28080/user/profile'); // 요청 경로

    try {
      var request = http.MultipartRequest('PUT', url);

      // 헤더 추가
      request.headers.addAll({
        'Authorization': 'Bearer $accessToken', // 사용자 access token 포함
      });

      // 닉네임 추가
      request.fields['nickname'] = _nicknameController.text;

      // 이미지 파일이 있는 경우 추가
      if (_profileImage != null) {
        request.files.add(await http.MultipartFile.fromPath(
          'profileImage',
          _profileImage!.path,
        ));
      }

      // 요청 보내기
      var response = await request.send();

      // 응답 확인
      if (response.statusCode == 200) {
        print('사용자 정보 수정 성공');
      } else {
        print("실패");
      }
    } catch (e) {
      print('네트워크 오류');
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Theme.of(context).scaffoldBackgroundColor, // 공유된 테마 사용
      body: Center(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 20.0),
          child: SingleChildScrollView(
            child: Column(
              // crossAxisAlignment: CrossAxisAlignment.center,
              children: [
                introPageHeader(
                    title: '', targetPage: FindPwScreen()), // targetPage 바꾸기
                Stack(
                  alignment: Alignment.bottomRight,
                  children: [
                    CircleAvatar(
                      radius: 50,
                      backgroundImage: _profileImage != null
                          ? FileImage(_profileImage!)
                          : AssetImage("assets/images/default_profile.jpg")
                              as ImageProvider,
                    ),
                    GestureDetector(
                      onTapDown: (details) =>
                          _showImagePicker(context, details), // 터치 위치 전달,
                      child: Container(
                        padding: EdgeInsets.all(6),
                        decoration: BoxDecoration(
                          color: Color(0xFFCF8A7A),
                          shape: BoxShape.circle,
                          border: Border.all(color: Colors.white, width: 2),
                        ),
                        child: Icon(Icons.camera_alt,
                            color: Colors.white, size: 20),
                      ),
                    ),
                  ],
                ),
                SizedBox(height: 10),
                SizedBox(
                  width: 450,
                  child: Column(
                    children: [
                      // 아이디 입력 필드(수정 불가)
                      Row(
                        children: [
                          const Text("아이디", style: TextStyle(fontSize: 16)),
                          const SizedBox(width: 20),
                          SizedBox(
                            width: 330,
                            child: TextField(
                              readOnly: true,
                              controller: _emailController,
                              decoration: InputDecoration(
                                enabledBorder: const UnderlineInputBorder(
                                  borderSide: BorderSide(color: Colors.grey),
                                ),
                                focusedBorder: const UnderlineInputBorder(
                                  borderSide: BorderSide(color: Colors.grey),
                                ),
                              ),
                            ),
                          ),
                        ],
                      ),
                      const SizedBox(height: 20),
                      // 닉네임 입력 필드(수정 가능)
                      Row(
                        children: [
                          const Text("닉네임", style: TextStyle(fontSize: 16)),
                          const SizedBox(width: 20),
                          Expanded(
                            // width: 330,
                            child: TextField(
                              controller: _nicknameController,
                              decoration: InputDecoration(
                                hintText: userName,
                                enabledBorder: const UnderlineInputBorder(
                                  borderSide: BorderSide(color: Colors.grey),
                                ),
                                focusedBorder: const UnderlineInputBorder(
                                  borderSide: BorderSide(color: Colors.black),
                                ),
                                suffixIcon: IconButton(
                                  icon: const Icon(Icons.close,
                                      color: Colors.grey),
                                  onPressed: () {
                                    setState(() {
                                      _nicknameController.clear();
                                      _isDuplicate = false;
                                    });
                                  },
                                ),
                              ),
                              onChanged: (text) {
                                setState(() {}); // 'x' 버튼 갱신
                              },
                            ),
                          ),
                          const SizedBox(width: 10),
                          // 중복 확인 버튼
                          ElevatedButton(
                            style: ElevatedButton.styleFrom(
                              shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(12.0),
                              ),
                              backgroundColor: Color(0xFFCF8A7A),
                              foregroundColor: Colors.white,
                              padding: const EdgeInsets.symmetric(
                                  horizontal: 13, vertical: 13),
                            ),
                            onPressed: _checkNickname,
                            child: const Text("중복확인"),
                          ),
                        ],
                      ),
                      // 중복된 닉네임 경고 메시지
                      if (_isDuplicate)
                        Align(
                          alignment: Alignment.centerLeft, // 왼쪽 정렬
                          child: const Padding(
                            padding: EdgeInsets.only(left: 70),
                            child: Text(
                              "이미 가입된 닉네임입니다.",
                              style: TextStyle(color: Colors.red, fontSize: 12),
                            ),
                          ),
                        ),
                      SizedBox(height: 30),
                    ],
                  ),
                ),
                SizedBox(
                  width: 300,
                  child: ElevatedButton(
                    style: ElevatedButton.styleFrom(
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(12.0),
                      ),
                      padding: EdgeInsets.symmetric(vertical: 16.0),
                      backgroundColor:
                          _isModified ? Color(0xFF424242) : Colors.grey,
                    ),
                    onPressed: _isModified ? changeUserProfile : null,
                    child: Center(
                      child: Text(
                        '회원정보 수정',
                        style: TextStyle(
                          fontSize: 15.0,
                          color: Colors.white,
                        ),
                      ),
                    ),
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
