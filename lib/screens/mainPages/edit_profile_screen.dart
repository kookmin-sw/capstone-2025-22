import 'dart:io'; // 파일 및 디렉토리 관련 기능 제공
import 'dart:convert'; // JSON 인코딩 및 디코딩을 위해 필요
import 'dart:typed_data'; // 바이트 데이터를 다루기 위해 필요
import 'package:flutter/material.dart'; // Flutter UI 요소 사용
import 'package:http/http.dart' as http; // HTTP 요청 위해 사용
import 'package:http_parser/http_parser.dart'; // HTTP 요청에서 파일 업로드 시 사용
import 'package:image_picker/image_picker.dart'; // 갤러리 및 카메라에서 이미지 선택
import 'package:path_provider/path_provider.dart'; // 앱의 저장 디렉토리 찾기
import 'package:flutter_secure_storage/flutter_secure_storage.dart'; // 안전한 로컬 저장소 사용
import 'package:capstone_2025/screens/mainPages/navigation_screens.dart';
import 'package:capstone_2025/screens/introPages/widgets/intro_page_header.dart';

/// 사용자 정보 수정하는 페이지
class EditProfileScreen extends StatefulWidget {
  const EditProfileScreen({super.key});

  @override
  State<EditProfileScreen> createState() => _EditProfileScreenState();
}

class _EditProfileScreenState extends State<EditProfileScreen> {
  // 사용자 정보 입력 필드 컨트롤러
  final TextEditingController _emailController =
      TextEditingController(); // 이메일 입력 필드 컨트롤러
  final TextEditingController _nicknameController =
      TextEditingController(); // 닉네임 입력 필드 컨트롤러

  // 보안 저장소
  final _storage = const FlutterSecureStorage();

  // 사용자 상태 정보
  bool _isDuplicate = false; // 닉네임 중복 여부
  bool _isModified = false; // 회원정보 수정 여부
  String? _message; // 중복 확인 후 메시지
  Color _messageColor = Colors.red; // 메시지 색상

  // secure storage에 저장된 사용자 정보
  String? email;
  String? nickName;
  String? accessToken; // 인증 토큰 (API 요청에 필요)
  File? _profileImage; // 프로필 사진

  // 앱이 실행되면 자동으로 secure storage에서 사용자 정보를 불러옴
  @override
  void initState() {
    super.initState();
    _loadUserData(); // secure storage에서 유저 데이터 불러오기
  }

  /// Secure Storage에서 데이터 불러와 UI 업데이트
  Future<void> _loadUserData() async {
    email = await _storage.read(key: 'user_email');
    nickName = await _storage.read(key: 'nick_name');
    accessToken = await _storage.read(key: 'access_token');
    String? profileImagePath = await _storage.read(key: 'profile_image');

    // UI 업데이트
    setState(() {
      _emailController.text = email ?? "example@gmail.com";
      _nicknameController.text = nickName ?? "홍길동";
      _profileImage =
          (profileImagePath != null && File(profileImagePath).existsSync())
              ? File(profileImagePath) // 저장된 이미지가 있으면 사용
              : null; // 없으면 null (UI에서 기본 이미지 표시)
    });
  }

  /// 중복 확인 메시지 업데이트 함수
  void _updateMessage(String message, Color color) {
    setState(() {
      _message = message;
      _messageColor = color;
    });
  }

  /// 닉네임 중복 검사 API 호출
  Future<void> _checkNickname() async {
    String nickname = _nicknameController.text.trim(); // 닉네임 앞뒤 공백 제거
    if (nickname.isEmpty) {
      _updateMessage("닉네임을 입력하세요.", Colors.red);
      return;
    }

    final uri = Uri.parse(
        'http://10.0.2.2:28080/verification/nicknames?nickname=${_nicknameController.text}');
    final response = await http.get(uri);
    final data = jsonDecode(response.body); // 응답을 JSON으로 변환
    print(data['body']);
    print(response.statusCode);

    setState(() => _message = null); // 기존 오류 메시지 초기화

    if (data['body'] == 'invalid') {
      _updateMessage("이미 가입된 닉네임입니다.", Colors.red);
      setState(() => _isDuplicate = true);
    } else {
      _updateMessage("사용 가능한 닉네임입니다", Colors.green);
      setState(() {
        _isDuplicate = false;
        _isModified = true; // 정보가 변경됨);
      });
    }
  }

  /// 프로필 사진 선택 (갤러리 또는 카메라)
  Future<void> _pickImage(ImageSource source) async {
    final picker = ImagePicker();
    final pickedFile = await picker.pickImage(source: source);
    if (pickedFile != null) {
      setState(() {
        _profileImage = File(pickedFile.path); // 선택한 이미지를 File로 변환
        _isModified = true; // 정보가 변경됨
      });
    }
  }

  /// 프로필 사진 변경 옵션 메뉴 표시
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
            title: const Text("사진 촬영"),
            onTap: () {
              Navigator.pop(context); // 메뉴 닫기
              _pickImage(ImageSource.camera);
            },
          ),
        ),
        PopupMenuItem(
          child: ListTile(
            title: const Text("앨범에서 선택"),
            onTap: () {
              Navigator.pop(context); // 메뉴 닫기
              _pickImage(ImageSource.gallery);
            },
          ),
        ),
      ],
    );
  }

  /// 사용자 정보 업데이트 API 호출 (닉네임, 프로필 사진)
  Future<void> _updateUserProfile() async {
    final url = Uri.parse('http://10.0.2.2:28080/users/profile'); // 요청 경로

    try {
      var request = http.MultipartRequest('PUT', url);
      request.headers.addAll({'authorization': 'Bearer $accessToken'});
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
        final data = jsonDecode(responseBody);
        await _saveUserData(data['body']); // Secure Storage에 사용자 정보 저장

        // 메인 화면으로 이동
        if (mounted) {
          Navigator.pushReplacement(
            context,
            MaterialPageRoute(builder: (context) => NavigationScreens()),
          );
        }
      }
    } catch (e) {
      print('네트워크 오류');
    }
  }

  /// 사용자 정보 저장 (secure storage)
  Future<void> _saveUserData(Map<String, dynamic> userData) async {
    await _storage.write(key: 'nick_name', value: userData['nickname']);

    // base64 이미지 변환 및 저장
    if (userData['profileImage'] != null) {
      try {
        Uint8List imageBytes =
            base64Decode(userData['profileImage']); // base64 디코딩
        Directory tempDir =
            await getApplicationDocumentsDirectory(); // 저장할 디렉토리
        String filePath = '${tempDir.path}/profile_image.png';

        File imageFile = File(filePath);
        await imageFile.writeAsBytes(imageBytes); // 파일로 저장
        await _storage.write(
            key: 'profile_image', value: filePath); // 저장된 경로 기록

        // UI 업데이트
        setState(() => _profileImage = imageFile);

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
        child: SingleChildScrollView(
          padding: const EdgeInsets.symmetric(horizontal: 20.0),
          child: Column(
            children: [
              introPageHeader(title: '', targetPage: NavigationScreens()),
              _buildProfileImage(),
              const SizedBox(height: 10),
              _buildUserInfoForm(),
              _buildUpdateButton(),
            ],
          ),
        ),
      ),
    );
  }

  /// 프로필 이미지 위젯
  Widget _buildProfileImage() {
    return Stack(
      // 여러 개의 위젯을 겹쳐서 배치하는 Stack 위젯 사용
      alignment: Alignment.bottomRight, // 카메라 아이콘 위치: 하단 오른쪽 정렬
      children: [
        CircleAvatar(
          // 원형 프로필 이미지
          radius: 50, // 원형 프로필 이미지 크기 (반지름 50)
          backgroundImage: _profileImage != null
              ? FileImage(_profileImage!) // 사용자가 설정한 프로필 이미지
              : const AssetImage("assets/images/default_profile.jpg")
                  as ImageProvider, // 기본 이미지
        ),
        GestureDetector(
          // 터치 이벤트 감지 (카메라 아이콘 클릭)
          onTapDown: (details) =>
              _showImagePicker(context, details), // 터치하면 사진 변경 메뉴 표시
          child: _buildCameraIcon(), // 카메라 아이콘
        ),
      ],
    );
  }

  /// 카메라 아이콘 버튼
  Widget _buildCameraIcon() {
    return Container(
      // 원형 버튼을 만들기 위해 container 사용
      padding: EdgeInsets.all(6), // 내부 여백 추가
      decoration: BoxDecoration(
        color: Color(0xFFCF8A7A), // 버튼 색
        shape: BoxShape.circle, // 원형 버튼 만들기
        border: Border.all(color: Colors.white, width: 2), // 흰색 테두리 추가
      ),
      child: const Icon(Icons.camera_alt,
          color: Colors.white, size: 20), // 카메라 아이콘
    );
  }

  /// 회원정보 수정 버튼
  Widget _buildUpdateButton() {
    return SizedBox(
      width: 300, // 버튼의 너비
      child: ElevatedButton(
        style: ElevatedButton.styleFrom(
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(12.0), // 버튼 모서리를 둥글게 설정
          ),
          padding: EdgeInsets.symmetric(vertical: 16.0), // 위아래 여백 추가
          backgroundColor: _isModified
              ? Color(0xFF424242)
              : Colors.grey, // 변경이 있으면 검정색, 없으면 회색
        ),
        onPressed:
            _isModified ? _updateUserProfile : null, // 변경된 정보가 있을 때만 버튼 활성화
        child: const Text(
          '회원정보 수정',
          style: TextStyle(fontSize: 15.0, color: Colors.white),
        ),
      ),
    );
  }

  /// 사용자 정보 입력 폼 (아이디, 닉네임, 중복확인 버튼)
  Widget _buildUserInfoForm() {
    return SizedBox(
      width: 450, // 전체 입력 폼의 너비 설정
      child: Column(
        children: [
          // 테이블 형식의 입력 필드 (아이디 & 닉네임)
          Table(
            columnWidths: const {
              0: IntrinsicColumnWidth(), // 첫 번째 열: 아이디, 닉네임 텍스트 (적절한 크기 자동 조절)
              1: FlexColumnWidth(), // 두 번째 열: 입력 필드 (유동적인 크기)
              2: IntrinsicColumnWidth(), // 세 번째 열: 버튼 (적절한 크기 자동 조절)
            },
            defaultVerticalAlignment:
                TableCellVerticalAlignment.middle, // 세로 중앙 정렬
            children: [
              // 아이디 입력 필드(수정 불가)
              TableRow(
                children: [
                  Padding(
                    padding: const EdgeInsets.only(right: 20, bottom: 15),
                    child: const Text("아이디", style: TextStyle(fontSize: 16)),
                  ),
                  Padding(
                    padding: const EdgeInsets.only(right: 10, bottom: 15),
                    child: SizedBox(
                      width: 300, // 입력 필드 크기
                      child: TextField(
                        readOnly: true, // 아이디는 수정 불가능
                        controller: _emailController,
                        decoration: InputDecoration(
                          enabledBorder: const UnderlineInputBorder(
                            borderSide: BorderSide(
                                color: Colors.grey), // 비활성 상태일 때 밑줄 색상
                          ),
                          focusedBorder: const UnderlineInputBorder(
                            borderSide: BorderSide(
                                color: Colors.grey), // 활성 상태일 때 밑줄 색상
                          ),
                        ),
                      ),
                    ),
                  ),
                  const SizedBox(), // 세 번째 열 빈 공간 유지
                ],
              ),
              // 닉네임 입력 필드 + 중복확인 버튼
              TableRow(
                children: [
                  Padding(
                    padding: const EdgeInsets.only(right: 20),
                    child: const Text("닉네임", style: TextStyle(fontSize: 16)),
                  ),
                  Padding(
                    padding: const EdgeInsets.only(right: 10),
                    child: SizedBox(
                      width: 300, // 입력 필드 크기 설정 (아이디 필드와 동일한 크기)
                      child: TextField(
                        controller: _nicknameController, // 닉네임 입력 필드
                        decoration: InputDecoration(
                          hintText: nickName, // 기존 닉네임 표시
                          hintStyle: TextStyle(color: Colors.grey), // 힌트 텍스트 색상
                          enabledBorder: const UnderlineInputBorder(
                            borderSide:
                                BorderSide(color: Colors.grey), // 비활성 상태 밑줄 색상
                          ),
                          focusedBorder: const UnderlineInputBorder(
                            borderSide:
                                BorderSide(color: Colors.black), // 활성 상태 밑줄 색상
                          ),
                          suffixIcon: IconButton(
                            icon: const Icon(Icons.close,
                                color: Colors.grey), // x 버튼
                            onPressed: () {
                              setState(() {
                                _nicknameController.clear(); // 입력된 닉네임 삭제
                                _isDuplicate = false;
                                _message = "닉네임을 입력하세요.";
                                _messageColor = Colors.red;
                              });
                            },
                          ),
                        ),
                        onChanged: (text) {
                          setState(() {}); // 텍스트 변경 시 UI 갱신
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
                                  BorderRadius.circular(12.0), // 버튼 모서리 둥글게
                            ),
                            backgroundColor: Color(0xFFCF8A7A), // 버튼 색상
                            foregroundColor: Colors.white, // 버튼 글자색
                            padding: const EdgeInsets.symmetric(
                                horizontal: 13, vertical: 13),
                          ),
                          onPressed: _checkNickname, // 중복 확인 버튼 클릭 시 실행
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
          const SizedBox(height: 20),
        ],
      ),
    );
  }

  /// 오류 메시지(닉네임 중복 확인 실패 시 표시)
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
