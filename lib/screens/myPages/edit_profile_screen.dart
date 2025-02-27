import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import 'package:capstone_2025/screens/introPages/login_screen.dart';
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

  @override
  void initState() {
    super.initState();
    _loadUserData(); // secure storage에서 유저 데이터 불러오기

    // 닉네임이나 이메일 변경 시 _isModified 상태 갱신
    _emailController.addListener(_checkModifincation);
    _nicknameController.addListener(_checkModifincation);
  }

// Secure Storage에서 데이터 불러와서 입력 필드 초기화
  Future<void> _loadUserData() async {
    String? email = await _storage.read(key: 'user_email');
    String? userName = await _storage.read(key: 'user_name');

    setState(() {
      _emailController.text = email ?? "example@gmail.com";
      _nicknameController.text = userName ?? "홍길동";
    });
  }

  void _checkModifincation() {
    setState(() {
      _isModified = (_emailController.text != "example@gmail.com" ||
          _nicknameController.text != "홍길동" ||
          _profileImage != null);
    });
  }

  // 닉네임 중복 확인
  void _checkNickname() {
    setState(() {
      _isDuplicate = _nicknameController.text == "홍길동";
    });
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

  // 프로필 사진 변경 다이얼로그
  void _showImagePicker() {
    showModalBottomSheet(
      context: context,
      builder: (context) {
        return SafeArea(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              ListTile(
                leading: Icon(Icons.add_a_photo_outlined),
                title: Text("사진 촬영"),
                onTap: () {
                  Navigator.pop(context);
                  _pickImage(ImageSource.camera);
                },
              ),
              ListTile(
                leading: Icon(Icons.photo),
                title: Text("앨범에서 선택"),
                onTap: () {
                  Navigator.pop(context);
                  _pickImage(ImageSource.gallery);
                },
              ),
            ],
          ),
        );
      },
    );
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
              crossAxisAlignment: CrossAxisAlignment.center,
              children: [
                introPageHeader(
                    title: '', targetPage: LoginScreen()), // targetPage 바꾸기
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
                      onTap: _showImagePicker,
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
                SizedBox(height: 20),
                SizedBox(
                  width: 450,
                  child: Column(
                    children: [
                      // 아이디 입력 필드
                      Row(
                        children: [
                          const Text("아이디", style: TextStyle(fontSize: 16)),
                          const SizedBox(width: 20),
                          SizedBox(
                            width: 330,
                            child: TextField(
                              controller: _emailController,
                              decoration: InputDecoration(
                                hintText: "example@gmail.com",
                                enabledBorder: const UnderlineInputBorder(
                                  borderSide: BorderSide(color: Colors.grey),
                                ),
                                focusedBorder: const UnderlineInputBorder(
                                  borderSide: BorderSide(color: Colors.black),
                                ),
                              ),
                            ),
                          ),
                        ],
                      ),
                      const SizedBox(height: 20),
                      // 닉네임 입력 필드
                      Row(
                        children: [
                          const Text("닉네임", style: TextStyle(fontSize: 16)),
                          const SizedBox(width: 20),
                          SizedBox(
                            width: 330,
                            child: TextField(
                              controller: _nicknameController,
                              decoration: InputDecoration(
                                hintText: "홍길동",
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
                    onPressed: _isModified
                        ? () {
                            // 함수 추가하기
                          }
                        : null,
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
