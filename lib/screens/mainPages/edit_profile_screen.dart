import 'dart:io';
import 'dart:convert';
import 'dart:typed_data';
import 'package:path_provider/path_provider.dart';
import 'package:capstone_2025/screens/mainPages/navigation_screens.dart';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:http_parser/http_parser.dart';
// ignore: depend_on_referenced_packages
import 'package:image_picker/image_picker.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';
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
  String? _message; // 중복 확인 후 메시지
  Color _messageColor = Colors.red;

  // secure storage에서 불러온 데이터 저장
  String? email;
  String? nickName;
  String? accessToken;

  @override
  void initState() {
    super.initState();
    _loadUserData(); // secure storage에서 유저 데이터 불러오기
  }

// Secure Storage에서 데이터 불러와서 필드 초기화
  Future<void> _loadUserData() async {
    email = await _storage.read(key: 'user_email');
    nickName = await _storage.read(key: 'nick_name');
    accessToken = await _storage.read(key: 'access_token');
    String? profileImagePath = await _storage.read(key: 'profile_image');

    setState(() {
      _emailController.text = email ?? "example@gmail.com";
      _nicknameController.text = nickName ?? "홍길동";

      // 프로필 이미지가 있으면 file 객체로 변환
      if (profileImagePath != null && profileImagePath.isNotEmpty) {
        _profileImage = File(profileImagePath);
      } else {
        _profileImage = null; // 새로운 사용자에게 기본 이미지 적용
      }
    });
  }

// 서버에서 닉네임 중복 확인
  Future<void> _checkNickname() async {
    // 닉네임을 입력하지 않았을 때
    if (_nicknameController.text == '') {
      setState(() {
        _message = "닉네임을 입력하세요.";
        _messageColor = Colors.red;
      });
      return;
    }

    final uri = Uri.parse(
        'http://10.0.2.2:28080/verification/nicknames?nickname=${_nicknameController.text}');
    final response = await http.get(uri);
    final data = jsonDecode(response.body);
    print(data['body']);
    print(response.statusCode);

    setState(() => _message = null); // 기존 오류 메시지 초기화

    if (data['body'] == 'invalid') {
      setState(() {
        _isDuplicate = true; // invalid이면 이미 존재하는 이메일
        _isModified = false;
        _message = "이미 가입된 닉네임입니다.";
        _messageColor = Colors.red;
      });
    } else {
      setState(() {
        _isModified = true;
        _message = "중복확인에 성공했습니다.";
        _messageColor = Colors.green;
      });
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

  Future<void> changeUserProfile({bool useBase64 = false}) async {
    print("Access Token: $accessToken"); // 토큰 값 출력
    final url = Uri.parse('http://10.0.2.2:28080/users/profile'); // 요청 경로

    try {
      var request = http.MultipartRequest('PUT', url);

      // 헤더 추가
      request.headers.addAll({
        'authorization': 'Bearer $accessToken', // 사용자 access token
        'accept': '*/*', // 요청 accept 타입 설정
      });

      // 닉네임 추가
      request.fields['nickname'] = _nicknameController.text;

      if (_profileImage != null) {
        request.files.add(await http.MultipartFile.fromPath(
          'profileImage',
          _profileImage!.path,
          contentType: MediaType('image', 'jpeg'),
        ));
      }

      // 요청 보내기
      var response = await request.send();
      var responseBody = await response.stream.bytesToString();

      print("응답 코드: ${response.statusCode}");
      print("응답 바디: $responseBody");

      // 응답 확인
      if (response.statusCode == 200) {
        print('사용자 정보 수정 성공');

        final data = jsonDecode(responseBody);
        await _saveUserData(data['body']); // Secure Storage에 사용자 정보 저장

        // 메인 화면으로 이동
        if (mounted) {
          Navigator.pushReplacement(
            context,
            MaterialPageRoute(builder: (context) => NavigationScreens()),
          );
        }
      } else {
        print("실패: ${response.reasonPhrase}");
      }
    } catch (e) {
      print('네트워크 오류');
    }
  }

  /// 사용자 정보 수정 성공 시 사용자 정보 저장
  Future<void> _saveUserData(Map<String, dynamic> userData) async {
    await _storage.write(key: 'nick_name', value: userData['nickname']);

    // base64 이미지 변환 및 저장
    if (userData['profileImage'] != null &&
        userData['profileImage'].isNotEmpty) {
      String base64Image = userData['profileImage'];

      try {
        Uint8List imageBytes = base64Decode(base64Image); // Base64 디코딩
        Directory tempDir =
            await getApplicationDocumentsDirectory(); // 저장할 디렉토리
        String filePath = '${tempDir.path}/profile_image.png';

        File imageFile = File(filePath);
        await imageFile.writeAsBytes(imageBytes); // 파일로 저장

        await _storage.write(
            key: 'profile_image', value: filePath); // 저장된 경로 기록

        setState(() {
          _profileImage = imageFile; // UI 업데이트
        });

        print("프로필 이미지 저장 완료: $filePath");
      } catch (e) {
        print("프로필 이미지 저장 실패: $e");
      }
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
                introPageHeader(title: '', targetPage: NavigationScreens()),
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
                      Table(
                        columnWidths: const {
                          0: IntrinsicColumnWidth(), // 첫 번째 열 (아이디 / 닉네임)
                          1: FlexColumnWidth(), // 두 번째 열 (입력 칸)
                          2: IntrinsicColumnWidth(), // 세 번째 열 (버튼)
                        },
                        defaultVerticalAlignment:
                            TableCellVerticalAlignment.middle, // 세로 정렬 맞추기
                        children: [
                          // 아이디 행
                          TableRow(
                            children: [
                              Padding(
                                padding: const EdgeInsets.only(
                                    right: 20, bottom: 15),
                                child: const Text("아이디",
                                    style: TextStyle(fontSize: 16)),
                              ),
                              Padding(
                                padding: const EdgeInsets.only(
                                    right: 10, bottom: 15),
                                child: SizedBox(
                                  width: 300,
                                  child: TextField(
                                    readOnly: true,
                                    controller: _emailController,
                                    decoration: InputDecoration(
                                      enabledBorder: const UnderlineInputBorder(
                                        borderSide:
                                            BorderSide(color: Colors.grey),
                                      ),
                                      focusedBorder: const UnderlineInputBorder(
                                        borderSide:
                                            BorderSide(color: Colors.grey),
                                      ),
                                    ),
                                  ),
                                ),
                              ),
                              const SizedBox(), // 빈 공간 유지
                            ],
                          ),
                          // 닉네임 행
                          TableRow(
                            children: [
                              Padding(
                                padding: const EdgeInsets.only(right: 20),
                                child: const Text("닉네임",
                                    style: TextStyle(fontSize: 16)),
                              ),
                              Padding(
                                padding: const EdgeInsets.only(right: 10),
                                child: SizedBox(
                                  width: 300, // 아이디 입력 칸과 동일한 크기 적용
                                  child: TextField(
                                    controller: _nicknameController,
                                    decoration: InputDecoration(
                                      hintText: nickName,
                                      hintStyle: TextStyle(
                                        color: Colors.grey,
                                      ),
                                      enabledBorder: const UnderlineInputBorder(
                                          borderSide:
                                              BorderSide(color: Colors.grey)),
                                      focusedBorder: const UnderlineInputBorder(
                                          borderSide:
                                              BorderSide(color: Colors.black)),
                                      suffixIcon: IconButton(
                                        icon: const Icon(Icons.close,
                                            color: Colors.grey),
                                        onPressed: () {
                                          setState(() {
                                            _nicknameController.clear();
                                            _isDuplicate = false;
                                            _message = "닉네임을 입력하세요.";
                                            _messageColor = Colors.red;
                                          });
                                        },
                                      ),
                                    ),
                                    onChanged: (text) {
                                      setState(() {}); // 'x' 버튼 갱신
                                    },
                                  ),
                                ),
                              ),
                              Padding(
                                padding: const EdgeInsets.only(left: 10),
                                child: Wrap(
                                  children: [
                                    ElevatedButton(
                                      style: ElevatedButton.styleFrom(
                                        shape: RoundedRectangleBorder(
                                          borderRadius:
                                              BorderRadius.circular(12.0),
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
                              ),
                            ],
                          ),
                        ],
                      ),
                      // 중복된 닉네임 경고 메시지
                      if (_message != null) buildErrorMessage(),
                      SizedBox(height: 20),
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

  /// 오류 메시지
  Widget buildErrorMessage() {
    return Align(
      alignment: Alignment.centerLeft, // 왼쪽 정렬
      child: Padding(
        padding: const EdgeInsets.only(left: 70),
        child: Text(
          _message!,
          style: TextStyle(color: _messageColor),
        ),
      ),
    );
  }
}
