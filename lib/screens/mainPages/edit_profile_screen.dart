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

  // secure storage
  final _storage = const FlutterSecureStorage();

  // Secure Storage 또는 api로부터 불러온 사용자 정보 저장하는 변수
  String? email;
  String? nickName;
  String? accessToken; // 인증 토큰 (API 요청에 필요)
  File? _profileImage; // 프로필 사진

  // 상태 정보
  bool _isModified = false; // 회원정보 수정 여부
  String _message = ""; // 중복 확인 후 메시지 텍스트
  Color _messageColor = Colors.red; // 메시지 텍스트 색상(오류: 빨강, 정상: 초록)

  // 옵션 메뉴 위치 계산 시 사용
  final GlobalKey _cameraIconKey = GlobalKey(); // 카메라 아이콘 위치 찾기 위함

  // 메모리 기반 이미지 저장 변수
  Uint8List? _profileImageBytes; // 서버에서 받아서 프로필 사진을 로컬에 저장하지 않고 화면에 바로 출력하기 위함

  // 상수
  static const String defaultEmail = "example@gmail.com";
  static const String defaultNickname = "홍길동";
  static const String profileImageName =
      'profile_image.png'; // 앱 내 임시 저장할 프로필 사진 파일 이름
  static const String defaultProfileImagePath =
      "assets/images/default_profile.jpg"; // 	프로필 이미지가 없을 때 보여줄 기본 이미지 경로
  static const String baseUrl = 'http://10.0.2.2:28080';

  /// 앱이 실행되면 자동으로 secure storage에서 사용자 정보를 불러옴
  @override
  void initState() {
    super.initState();
    _loadUserData(); // secure storage에서 유저 데이터 불러오기
  }

  /// Secure Storage 또는 api로부터 데이터 불러와 UI 업데이트
  Future<void> _loadUserData() async {
    email = await _storage.read(key: 'user_email');
    nickName = await _storage.read(key: 'nick_name');
    accessToken = await _storage.read(key: 'access_token');

    // UI 업데이트
    setState(() {
      _emailController.text = email ?? defaultEmail;
      _nicknameController.text = nickName ?? defaultNickname;
    });

    if (email != null && accessToken != null) {
      await _fetchUserProfile();
    }
  }

  /// 서버에서 프로필 사진 가져오는 함수(api 사용)
  Future<void> _fetchUserProfile() async {
    try {
      final uri = Uri.parse('$baseUrl/user/email?email=$email');
      final response = await http.get(
        uri,
        headers: {'authorization': 'Bearer $accessToken'},
      );

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        final body = data['body'];
        await _handleProfileImage(body['profileImage']);
      }
    } catch (e) {
      print("유저 정보 로딩 실패: $e");
    }
  }

  /// 프로필 이미지 처리 (파일 저장 없이 메모리에서 직접 보여줌)
  Future<void> _handleProfileImage(String? profileImageBase64) async {
    if (profileImageBase64 != null) {
      Uint8List imageBytes = base64Decode(profileImageBase64);

      setState(() {
        _profileImage = null; // File은 사용하지 않음
        _profileImageBytes = imageBytes;
      });
    } else {
      print("프로필 사진이 없음.");
    }
  }

  /// 중복 확인 메시지 업데이트 함수
  void _updateMessage(String message, Color color) {
    setState(() {
      _message = message; // 메시지 텍스트
      _messageColor = color; // 메시지 텍스트 색상
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
        '$baseUrl/verification/nicknames?nickname=${_nicknameController.text}');
    final response = await http.get(uri);
    final data = jsonDecode(response.body); // 응답을 JSON으로 변환
    print(data['body']);
    print(response.statusCode);

    setState(() => _message = ""); // 기존 오류 메시지 초기화

    if (response.statusCode == 200) {
      if (data['body'] == 'invalid') {
        _updateMessage("이미 가입된 닉네임입니다.", Colors.red);
      } else {
        _updateMessage("사용 가능한 닉네임입니다", Colors.green);
        setState(() => _isModified = true); // 정보가 변경됨
      }
    } else {
      _updateMessage("서버 오류가 발생했습니다. 잠시 후 다시 시도해주세요.", Colors.red);
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
  void _showImagePicker(BuildContext context) {
    final RenderBox overlay =
        Overlay.of(context).context.findRenderObject() as RenderBox;
    final RenderBox buttonRenderBox =
        _cameraIconKey.currentContext!.findRenderObject() as RenderBox;

    final Offset buttonPosition =
        buttonRenderBox.localToGlobal(Offset.zero, ancestor: overlay);
    final Size buttonSize = buttonRenderBox.size;

    final double screenWidth = MediaQuery.of(context).size.width;
    final double screenHeight = MediaQuery.of(context).size.height;

    final double offsetX = screenWidth * 0.24; // 화면 너비의 24%
    final double offsetY = screenHeight * 0.01; // 화면 높이의 1%

    final RelativeRect position = RelativeRect.fromLTRB(
      buttonPosition.dx + buttonSize.width + offsetX,
      buttonPosition.dy + buttonSize.height + offsetY,
      screenWidth - (buttonPosition.dx + offsetX),
      screenHeight - (buttonPosition.dy + buttonSize.height + offsetY),
    );

    showMenu(
      context: context,
      position: position,
      items: [
        PopupMenuItem(
          child: GestureDetector(
            onTap: () {
              Navigator.pop(context);
              _pickImage(ImageSource.camera);
            },
            child: _buildMenuItem("사진 촬영", Icons.photo_camera_outlined),
          ),
        ),
        PopupMenuItem(
          padding: EdgeInsets.all(0),
          enabled: false, // 클릭 안 되게
          height: 1,
          child: Divider(thickness: 1, height: 1, indent: 0, endIndent: 0),
        ),
        PopupMenuItem(
          child: GestureDetector(
            onTap: () {
              Navigator.pop(context);
              _pickImage(ImageSource.gallery);
            },
            child: _buildMenuItem("앨범에서 선택", Icons.image_outlined),
          ),
        ),
      ],
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(16), // 둥근 테두리
      ),
      color: Colors.white,
    );
  }

  Widget _buildMenuItem(String text, IconData icon) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        Text(text),
        Icon(icon, color: Colors.black54),
      ],
    );
  }

  /// 사용자 정보 업데이트 API 호출 (닉네임, 프로필 사진)
  Future<void> _updateUserProfile() async {
    final url = Uri.parse('$baseUrl/user/profile'); // 요청 경로

    try {
      var request = http.MultipartRequest('PUT', url);
      request.headers['authorization'] = 'Bearer $accessToken';
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
      } else {
        print('업데이트 실패: ${response.statusCode}');
      }
    } catch (e) {
      print('네트워크 오류');
    }
  }

  /// 사용자 정보 저장 (secure storage)
  Future<void> _saveUserData(Map<String, dynamic> userData) async {
    await _storage.write(key: 'nick_name', value: userData['nickname']);
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
              ? MemoryImage(_profileImageBytes!) // 메모리에서 바로 이미지 표시
              : const AssetImage(defaultProfileImagePath)
                  as ImageProvider, // 기본 이미지
        ),
        GestureDetector(
          // 터치 이벤트 감지 (카메라 아이콘 클릭)
          onTap: () => _showImagePicker(context), // 터치하면 사진 변경 메뉴 표시
          child: _buildCameraIcon(), // 카메라 아이콘
        ),
      ],
    );
  }

  /// 카메라 아이콘 버튼 ui
  Widget _buildCameraIcon() {
    return Container(
      key: _cameraIconKey,
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

  /// 사용자 정보 입력 폼 (아이디, 닉네임, 중복확인 버튼)
  Widget _buildUserInfoForm() {
    return SizedBox(
      width: 450, // 전체 입력 폼의 너비 설정
      child: Column(
        children: [
          // 테이블 형식의 입력 필드 (아이디 & 닉네임)
          _buildUserTable(),
          // 중복된 닉네임 경고 메시지
          _buildErrorMessage(),
          const SizedBox(height: 20),
        ],
      ),
    );
  }

  Table _buildUserTable() {
    return Table(
      columnWidths: const {
        0: IntrinsicColumnWidth(), // 첫 번째 열: 아이디, 닉네임 텍스트 (적절한 크기 자동 조절)
        1: FlexColumnWidth(), // 두 번째 열: 입력 필드 (유동적인 크기)
        2: IntrinsicColumnWidth(), // 세 번째 열: 버튼 (적절한 크기 자동 조절)
      },
      defaultVerticalAlignment: TableCellVerticalAlignment.middle, // 세로 중앙 정렬
      children: [
        // 아이디 입력 필드(수정 불가)
        _buildEmailRow(),
        // 닉네임 입력 필드 + 중복확인 버튼
        _buildNicknameRow(),
      ],
    );
  }

  TableRow _buildEmailRow() {
    return TableRow(
      children: [
        const Padding(
          padding: EdgeInsets.only(right: 20, bottom: 15),
          child: Text("아이디", style: TextStyle(fontSize: 16)),
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
                  borderSide: BorderSide(color: Colors.grey), // 비활성 상태일 때 밑줄 색상
                ),
                focusedBorder: const UnderlineInputBorder(
                  borderSide: BorderSide(color: Colors.grey), // 활성 상태일 때 밑줄 색상
                ),
              ),
            ),
          ),
        ),
        const SizedBox(), // 세 번째 열 빈 공간 유지
      ],
    );
  }

  TableRow _buildNicknameRow() {
    return TableRow(
      children: [
        const Padding(
          padding: EdgeInsets.only(right: 20),
          child: Text("닉네임", style: TextStyle(fontSize: 16)),
        ),
        Padding(
          padding: const EdgeInsets.only(right: 10),
          child: SizedBox(
            width: 300, // 입력 필드 크기 설정 (아이디 필드와 동일한 크기)
            child: TextField(
              controller: _nicknameController, // 닉네임 입력 필드
              decoration: InputDecoration(
                hintText: nickName, // 기존 닉네임 표시
                hintStyle: const TextStyle(color: Colors.grey), // 힌트 텍스트 색상
                enabledBorder: const UnderlineInputBorder(
                  borderSide: BorderSide(color: Colors.grey), // 비활성 상태 밑줄 색상
                ),
                focusedBorder: const UnderlineInputBorder(
                  borderSide: BorderSide(color: Colors.black), // 활성 상태 밑줄 색상
                ),
                suffixIcon: IconButton(
                  icon: const Icon(Icons.close, color: Colors.grey), // x 버튼
                  onPressed: () {
                    setState(() {
                      _nicknameController.clear(); // 입력된 닉네임 삭제
                      _message = "닉네임을 입력하세요.";
                      _messageColor = Colors.red;
                    });
                  },
                ),
              ),
              onChanged: (text) => setState(() {}), // 텍스트 변경 시 UI 갱신
            ),
          ),
        ),
        Padding(
          padding: const EdgeInsets.only(left: 10),
          child: ElevatedButton(
            style: ElevatedButton.styleFrom(
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(12.0), // 버튼 모서리 둥글게
              ),
              backgroundColor: const Color(0xFFCF8A7A), // 버튼 색상
              foregroundColor: Colors.white, // 버튼 글자색
              padding: const EdgeInsets.symmetric(horizontal: 13, vertical: 13),
            ),
            onPressed: _checkNickname, // 중복 확인 버튼 클릭 시 실행
            child: const Text("중복확인"),
          ),
        ),
      ],
    );
  }

  /// 오류 메시지(닉네임 중복 확인 실패 시 표시)
  Widget _buildErrorMessage() {
    return Align(
      alignment: Alignment.centerLeft, // 왼쪽 정렬
      child: Padding(
        padding: const EdgeInsets.only(left: 70),
        child: Text(
          _message,
          style: TextStyle(color: _messageColor),
        ),
      ),
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
}
