import 'dart:convert';
import 'dart:io';
import 'package:capstone_2025/screens/drumSheetPages/drum_sheet_player.dart';
import 'package:capstone_2025/services/api_func.dart';
import 'package:capstone_2025/services/storage_service.dart';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:flutter_screenutil/flutter_screenutil.dart';
import '../../models/sheet.dart';
import 'widgets/sheet_card.dart';
import 'widgets/add_sheet_dialog.dart';
import 'package:http_parser/http_parser.dart';

void main() => runApp(const DrumSheetScreen());

class DrumSheetScreen extends StatelessWidget {
  const DrumSheetScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: '악보 목록',
      theme: ThemeData(
        fontFamily: 'Pretendard',
        scaffoldBackgroundColor: const Color(0xFFf5f5f5),
      ),
      home: const SheetListScreen(),
    );
  }
}

// 정렬 기준 옵션 enum
enum SortOption { date, name, recentPractice }

class SheetListScreen extends StatefulWidget {
  const SheetListScreen({super.key});

  @override
  State<SheetListScreen> createState() => _SheetListScreenState();
}

class _SheetListScreenState extends State<SheetListScreen> {
  SortOption _selectedSort = SortOption.date;
  bool _isSelectionMode = false;
  bool _isAllSelected = false;
  bool _isSearchMode = false;
  final TextEditingController _searchController = TextEditingController();
  List<Sheet> _searchResults = [];
  String? userEmail;
  String? accessToken;

  @override
  void initState() {
    super.initState();
    createSheetList();
  }

  // 악보 리스트 샘플 데이터
  final List<Sheet> _sheets = [];

  // API 관련 메서드들
  //악보 리스트 받아오기
  void createSheetList() async {
    String? email = await storage.read(key: 'user_email');
    String? token = await storage.read(key: 'access_token');
    final response = await getHTTP('/sheets', {'email': email});

    if (response['errMessage'] == null) {
      var body = response['body']['sheets'];

      if (!mounted) return;
      setState(() {
        userEmail = email;
        accessToken = token;
        _sheets.clear();
        _sheets
            .addAll(body.map<Sheet>((json) => Sheet.fromJson(json)).toList());
      });
    }
  }

  Future<List<Sheet>> fetchSheets() async {
    final response =
        await http.get(Uri.parse('http://34.68.164.98:28080/sheets'));
    if (response.statusCode == 200) {
      List jsonResponse = json.decode(response.body);
      return jsonResponse.map((sheet) => Sheet.fromJson(sheet)).toList();
    } else {
      throw Exception('악보 로딩에 실패했습니다');
    }
  }

  Future<String> convertFileToBase64(String filePath) async {
    final bytes = await File(filePath).readAsBytes();
    return base64Encode(bytes);
  }

  Future<Sheet> addSheet(String title, String artist, String filePath) async {
    if (userEmail == null) {
      userEmail = await storage.read(key: 'user_email');
      if (userEmail == null) {
        throw Exception("userEmail이 없습니다.");
      }
    }

    final uri = Uri.parse('http://34.68.164.98:28080/sheets');
    final request = http.MultipartRequest('POST', uri);

    // 헤더 확인
    request.headers['Authorization'] = 'Bearer $accessToken';

    // 메타 정보 JSON 준비
    final sheetMeta = jsonEncode({
      'sheetName': title,
      'artistName': artist,
      'color': '#BEBEBE',
      'userEmail': userEmail,
      'fileExtension': 'pdf',
      'owner': true,
    });
    // JSON 메타 정보 첨부
    request.files.add(
      http.MultipartFile.fromString(
        'sheetCreateMeta',
        sheetMeta,
        contentType: MediaType('application', 'json'),
      ),
    );

    // PDF 파일 첨부
    request.files.add(
      await http.MultipartFile.fromPath(
        'sheetFile',
        filePath,
        contentType: MediaType('application', 'pdf'),
      ),
    );

    print('파일 업로드 시작...');
    final streamedResponse = await request.send();
    final response = await http.Response.fromStream(streamedResponse);

    print('응답 코드: ${response.statusCode}');
    print('응답 바디: ${response.body}');

    if (response.statusCode == 200) {
      print('파일 업로드 성공');
      final Map<String, dynamic> decoded = json.decode(response.body);
      final body = decoded['body'];
      try {
        body['sheetName'] =
            utf8.decode(body['sheetName'].toString().codeUnits); // 문자열 디코딩
      } catch (e) {
        print('문자열 디코딩 실패: $e');
      }
      return Sheet.fromJson(body);
    } else {
      throw Exception('파일 업로드 실패 - 상태코드: ${response.statusCode}');
    }
  }

  Future<void> updateSheet(int id, String title) async {
    final response = await http.put(
      Uri.parse('http://34.68.164.98:28080/sheets/$id'),
      headers: <String, String>{
        'Content-Type': 'application/json; charset=UTF-8',
      },
      body: jsonEncode(<String, String>{'title': title}),
    );

    if (response.statusCode != 200) {
      throw Exception('Failed to update sheet');
    }
  }

  // 정렬 관련 메서드들
  List<Sheet> get _sortedSheets {
    List<Sheet> sorted = List.from(_sheets);
    switch (_selectedSort) {
      case SortOption.date:
        sorted.sort((a, b) => b.createdDate.compareTo(a.createdDate));
        break;
      case SortOption.name:
        sorted.sort((a, b) => a.title.compareTo(b.title));
        break;
      case SortOption.recentPractice:
        sorted
            .sort((a, b) => b.lastPracticedDate.compareTo(a.lastPracticedDate));
        break;
    }
    return sorted;
  }

  String get _sortLabel {
    switch (_selectedSort) {
      case SortOption.date:
        return '생성 날짜 순';
      case SortOption.name:
        return '이름 순';
      case SortOption.recentPractice:
        return '최근 연습한 순';
    }
  }

  void _onSortSelected(SortOption option) {
    setState(() {
      _selectedSort = option;
    });
  }

  // 선택 모드 관련 메서드들
  void _toggleSelectionMode() {
    setState(() {
      _isSelectionMode = !_isSelectionMode;
      if (!_isSelectionMode) {
        for (var sheet in _sheets) {
          sheet.isSelected = false;
        }
      }
    });
  }

  void _toggleSelectAll() {
    setState(() {
      _isAllSelected = !_isAllSelected;
      for (var sheet in _sheets) {
        sheet.isSelected = _isAllSelected;
      }
    });
  }

  void _onSheetSelected(Sheet sheet) {
    setState(() {
      sheet.isSelected = !sheet.isSelected;
    });
  }

  // 악보 관리 관련 메서드들
  void _renameSheet(Sheet sheet) {
    final TextEditingController controller =
        TextEditingController(text: sheet.title);

    showDialog(
      context: context,
      barrierDismissible: true,
      builder: (context) {
        return AlertDialog(
          title: const Text('악보 이름 수정'),
          content: TextField(
            controller: controller,
            decoration: const InputDecoration(
              hintText: '새 이름을 입력하세요',
            ),
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.of(context).pop(),
              child: const Text('취소'),
            ),
            ElevatedButton(
              onPressed: () async {
                var requestBody = {
                  'email': userEmail,
                  "color":
                      '#${sheet.color.value.toRadixString(16).padLeft(8, '0').substring(2).toUpperCase()}',
                  "name": controller.text,
                };
                var response = await putHTTP(
                    "/sheets/${sheet.sheetId}/name", {}, requestBody,
                    reqHeader: {
                      "Authorization": 'Bearer $accessToken',
                    });
                if (response['errMessage'] == null) {
                  setState(() {
                    sheet.title = controller.text;
                    _isSelectionMode = false;
                    _isAllSelected = false;
                    for (var s in _sheets) {
                      s.isSelected = false;
                    }
                  });
                } else {
                  print(response['errMessage']);
                }
                Navigator.of(context).pop();
              },
              child: const Text('저장'),
            ),
          ],
        );
      },
    );
  }

  final List<Color> customColors = [
    Color(0xFFBEBEBE),
    Color(0xFFF4B3B3),
    Color(0xFFF4DDB3),
    Color(0xFFb3f4b5),
    Color(0xFFb3eaf4),
    Color(0xFFdcb3f4),
  ];

  final Map<Color, Color> _iconColorMap = {
    Color(0xFFBEBEBE): Color(0xFF646464),
    Color(0xFFF4B3B3): Color(0xFFD16F6F),
    Color(0xFFF4DDB3): Color(0xFFD1A36F),
    Color(0xFFb3f4b5): Color(0xFF6FD172),
    Color(0xFFb3eaf4): Color(0xFF6FB4D1),
    Color(0xFFdcb3f4): Color(0xFFB96FD1),
  };

  void _changeSheetColor(Sheet sheet) {
    showDialog(
      context: context,
      barrierColor: Colors.transparent,
      builder: (context) {
        return Stack(
          children: [
            Positioned(
              bottom: 100.h,
              left: MediaQuery.of(context).size.width / 2 -
                  (customColors.length * 24) / 2 +
                  100,
              child: Material(
                color: Colors.transparent,
                child: Container(
                  padding: EdgeInsets.symmetric(horizontal: 5.w, vertical: 8.h),
                  decoration: BoxDecoration(
                    color: Colors.white,
                    borderRadius: BorderRadius.circular(24),
                    boxShadow: [
                      BoxShadow(
                        color: Colors.black.withOpacity(0.1),
                        blurRadius: 8,
                      ),
                    ],
                  ),
                  child: Row(
                    mainAxisSize: MainAxisSize.min,
                    children: customColors.map((color) {
                      return GestureDetector(
                        onTap: () async {
                          // 로딩바 출력
                          showDialog(
                            context: context,
                            barrierDismissible: false,
                            builder: (_) => const Dialog(
                              backgroundColor: Colors.transparent,
                              child: Center(
                                child: CircularProgressIndicator(),
                              ),
                            ),
                          );
                          var requestBody = {
                            'email': userEmail,
                            "color":
                                '#${color.value.toRadixString(16).padLeft(8, '0').substring(2).toUpperCase()}',
                            "name": sheet.title
                          };
                          var response = await putHTTP(
                              "/sheets/${sheet.sheetId}/color", {}, requestBody,
                              reqHeader: {
                                "Authorization": 'Bearer $accessToken',
                              });
                          if (response['errMessage'] == null) {
                            setState(() {
                              sheet.color = color;
                              _isSelectionMode = false;
                              _isAllSelected = false;
                              for (var s in _sheets) {
                                s.isSelected = false;
                              }
                            });
                          }
                          Navigator.of(context).pop();
                          Navigator.of(context).pop();
                        },
                        child: Container(
                          width: 10.w,
                          height: 30.h,
                          margin: EdgeInsets.symmetric(horizontal: 2.w),
                          decoration: BoxDecoration(
                            color: color,
                            shape: BoxShape.circle,
                            border: Border.all(color: Colors.white, width: 2),
                          ),
                        ),
                      );
                    }).toList(),
                  ),
                ),
              ),
            ),
          ],
        );
      },
    );
  }

  List<int> selectedSheetsId(List<Sheet> sheets) {
    return sheets
        .where((sheet) => sheet.isSelected)
        .map((sheet) => sheet.sheetId)
        .toList();
  }

  void _confirmDeleteMultiple(List<Sheet> sheets) async {
    showDialog(
      context: context,
      barrierDismissible: true,
      builder: (context) {
        return Dialog(
          backgroundColor: Colors.white,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(16),
          ),
          child: SizedBox(
            width: 100.w,
            child: Padding(
              padding: EdgeInsets.symmetric(vertical: 25.h, horizontal: 5.w),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Text(
                    '선택한 악보를 삭제하시겠습니까?',
                    style: TextStyle(
                      fontSize: 5.5.sp,
                      fontWeight: FontWeight.w600,
                      color: Color(0xFF646464),
                    ),
                  ),
                  SizedBox(height: 27.h),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                    children: [
                      ElevatedButton(
                        style: ElevatedButton.styleFrom(
                          backgroundColor: const Color(0xFFF2F2F2),
                          minimumSize: Size(40.w, 55.h),
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(12),
                          ),
                        ),
                        onPressed: () => Navigator.of(context).pop(),
                        child: const Text(
                          '취소',
                          style: TextStyle(
                            color: Color(0xFF646464),
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                      ),
                      ElevatedButton(
                        style: ElevatedButton.styleFrom(
                          backgroundColor: const Color(0xFFD97D6C),
                          minimumSize: Size(40.w, 55.h),
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(12),
                          ),
                        ),
                        onPressed: () async {
                          // 로딩바 출력
                          showDialog(
                            context: context,
                            barrierDismissible: false,
                            builder: (_) => const Dialog(
                              backgroundColor: Colors.transparent,
                              child: Center(
                                child: CircularProgressIndicator(),
                              ),
                            ),
                          );
                          var response = await deleteHTTP(
                              '/sheets', selectedSheetsId(_sheets));

                          if (response['errMessage'] == null) {
                            setState(() {
                              _sheets.removeWhere(
                                  (sheet) => sheets.contains(sheet));
                            });
                            Navigator.of(context).pop(); // 로딩바 닫기
                            Navigator.of(context).pop(); // 확인 다이얼로그 닫기
                            _toggleSelectionMode(); // Exit selection mode
                          } else {
                            print(response['errMessage']);
                            Navigator.of(context).pop(); // 로딩바 닫기
                            Navigator.of(context).pop(); // 확인 다이얼로그 닫기
                          }
                        },
                        child: const Text(
                          '확인',
                          style: TextStyle(
                            color: Colors.white,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                      ),
                    ],
                  )
                ],
              ),
            ),
          ),
        );
      },
    );
  }

  void _confirmPracticeStart(Sheet sheet) {
    showDialog(
      context: context,
      barrierDismissible: true,
      builder: (context) {
        return Dialog(
          backgroundColor: Colors.white,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(16),
          ),
          child: SizedBox(
            width: 100.w,
            child: Padding(
              padding: EdgeInsets.symmetric(vertical: 20.h, horizontal: 5.w),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Text(
                    '${sheet.title} - ${sheet.artistName}',
                    textAlign: TextAlign.center,
                    style: TextStyle(
                      fontSize: 7.sp,
                      fontWeight: FontWeight.w600,
                      color: Color(0xFF646464),
                    ),
                  ),
                  SizedBox(height: 10.h),
                  Text(
                    '연주를 시작하시겠습니까?',
                    style: TextStyle(
                      fontSize: 5.5.sp,
                      color: Color(0xFF646464),
                    ),
                  ),
                  SizedBox(height: 30.h),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                    children: [
                      ElevatedButton(
                        style: ElevatedButton.styleFrom(
                          backgroundColor: const Color(0xFFF2F2F2),
                          minimumSize: Size(40.w, 55.h),
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(12),
                          ),
                        ),
                        onPressed: () => Navigator.of(context).pop(),
                        child: const Text(
                          '취소',
                          style: TextStyle(
                            color: Color(0xFF646464),
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                      ),
                      ElevatedButton(
                        style: ElevatedButton.styleFrom(
                          backgroundColor: const Color(0xFFD97D6C),
                          minimumSize: Size(40.w, 55.h),
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(12),
                          ),
                        ),
                        onPressed: () async {
                          // 로딩화면 출력
                          showDialog(
                            context: context,
                            barrierDismissible: false,
                            builder: (_) => const Dialog(
                              backgroundColor: Colors.transparent,
                              child: Center(
                                child: CircularProgressIndicator(),
                              ),
                            ),
                          );
                          // 연주 시작 페이지로 이동
                          final response =
                              await getHTTP('/sheets/${sheet.sheetId}', {});
                          print("sheetID: ${sheet.sheetId}");

                          if (response['body']['sheetInfo'] == null) {
                            Navigator.of(context).pop(); // 로딩바 닫기
                            Navigator.of(context).pop(); // 확인 다이얼로그 닫기
                            ScaffoldMessenger.of(context).showSnackBar(
                              SnackBar(
                                content: Text(
                                  '악보 변환 중입니다. 잠시 후 다시 시도해주세요.',
                                  style: TextStyle(
                                      fontSize: 5.5.sp,
                                      color: Colors.white,
                                      fontWeight: FontWeight.w600),
                                ),
                                backgroundColor: Color(0xFFD97D6C),
                                duration: Duration(seconds: 2),
                                behavior: SnackBarBehavior.floating,
                                shape: RoundedRectangleBorder(
                                  borderRadius: BorderRadius.circular(12),
                                ),
                                margin: EdgeInsets.all(16.h),
                              ),
                            );
                          } else {
                            Navigator.of(context).pop(); // 로딩바 닫기
                            Navigator.of(context).pop(); // 확인 다이얼로그 닫기

                            Navigator.push(
                              context,
                              MaterialPageRoute(
                                builder: (context) => DrumSheetPlayer(
                                  sheetId: sheet.sheetId ?? 0,
                                  title: sheet.title,
                                  artist: sheet.artistName,
                                  sheetXmlData: response['body']['sheetInfo'],
                                ),
                              ),
                            );
                          }
                        },
                        child: const Text(
                          '확인',
                          style: TextStyle(
                            color: Colors.white,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                      ),
                    ],
                  )
                ],
              ),
            ),
          ),
        );
      },
    );
  }

  // 검색 기능 구현
  void _performSearch(String query) {
    setState(() {
      if (query.isEmpty) {
        _searchResults = [];
        return;
      }
      _searchResults = _sheets
          .where((sheet) =>
              sheet.title.toLowerCase().contains(query.toLowerCase()))
          .toList();
    });
  }

  // 검색 모드 토글
  void _toggleSearchMode() {
    setState(() {
      _isSearchMode = !_isSearchMode;
      if (!_isSearchMode) {
        _searchController.clear();
        _searchResults = [];
      }
    });
  }

  @override
  void dispose() {
    _searchController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final displayedSheets = _isSearchMode && _searchController.text.isNotEmpty
        ? _searchResults
        : _sortedSheets;
    final selectedSheets = _sheets.where((sheet) => sheet.isSelected).toList();

    return Scaffold(
      appBar: AppBar(
        backgroundColor: Color(0xFFF5F5F5),
        elevation: 0,
        titleSpacing: 10.w,
        toolbarHeight: 90.h,
        title: Padding(
          padding: EdgeInsets.only(top: 8.h),
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.center,
            children: [
              Icon(
                Icons.library_music,
                color: const Color(0xFF646464),
                size: 9.sp,
              ),
              SizedBox(width: 5.w),
              Text(
                '악보 목록',
                style: TextStyle(
                    color: const Color(0xFF646464),
                    fontSize: 9.sp,
                    fontWeight: FontWeight.w800),
              ),
            ],
          ),
        ),
        actions: [
          if (_isSelectionMode) ...[
            Container(
              margin: EdgeInsets.only(top: 8.h, right: 5.w),
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(32),
                boxShadow: [
                  BoxShadow(
                    color: const Color(0xFFd9d9d9),
                    blurRadius: 2,
                    offset: const Offset(0, 2),
                  ),
                ],
              ),
              child: TextButton(
                onPressed: _toggleSelectAll,
                style: TextButton.styleFrom(
                  padding: EdgeInsets.symmetric(horizontal: 8.w, vertical: 8.h),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(32),
                  ),
                ),
                child: Text(
                  _isAllSelected ? '전체 선택 해제' : '전체 선택',
                  style: TextStyle(
                      fontSize: 5.5.sp,
                      fontWeight: FontWeight.bold,
                      color: const Color(0xFF646464)),
                ),
              ),
            ),
            Container(
              margin: EdgeInsets.only(top: 8.h, right: 8.w),
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(32),
                boxShadow: [
                  BoxShadow(
                    color: const Color(0xFFd9d9d9),
                    blurRadius: 2,
                    offset: const Offset(0, 2),
                  ),
                ],
              ),
              child: TextButton(
                onPressed: _toggleSelectionMode,
                style: TextButton.styleFrom(
                  padding: EdgeInsets.symmetric(horizontal: 8.w, vertical: 8.h),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(32),
                  ),
                ),
                child: Text(
                  '완료',
                  style: TextStyle(
                      fontSize: 5.5.sp,
                      fontWeight: FontWeight.bold,
                      color: const Color(0xFF646464)),
                ),
              ),
            ),
          ] else if (_isSearchMode) ...[
            Container(
              width: 100.w,
              height: 50.h,
              margin: EdgeInsets.only(top: 8.h, right: 24.w),
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(10),
                boxShadow: [
                  BoxShadow(
                    color: Colors.black.withValues(alpha: 0.08),
                    blurRadius: 2,
                    offset: const Offset(0, 1),
                  ),
                ],
              ),
              child: Row(
                children: [
                  Padding(
                    padding: EdgeInsets.only(left: 5.w),
                    child: Icon(
                      Icons.search,
                      size: 6.5.sp,
                      color: Color(0xFF595959),
                    ),
                  ),
                  Expanded(
                    child: TextField(
                      controller: _searchController,
                      autofocus: true,
                      textAlignVertical: TextAlignVertical.center,
                      decoration: InputDecoration(
                        hintText: '검색어를 입력해주세요',
                        hintStyle: TextStyle(
                          color: Color(0xFFBEBEBE),
                          fontSize: 5.sp,
                        ),
                        border: InputBorder.none,
                        isDense: true,
                        contentPadding: EdgeInsets.symmetric(
                            horizontal: 2.w, vertical: 11.5.h),
                      ),
                      style: TextStyle(
                        color: Color(0xFF595959),
                        fontSize: 5.sp,
                      ),
                      onChanged: _performSearch,
                    ),
                  ),
                  IconButton(
                    padding: EdgeInsets.all(4.w),
                    constraints: const BoxConstraints(),
                    icon: Icon(
                      Icons.close,
                      size: 6.sp,
                      color: Color(0xFF595959),
                    ),
                    onPressed: _toggleSearchMode,
                  ),
                  SizedBox(width: 1.w),
                ],
              ),
            ),
          ] else ...[
            Padding(
              padding: EdgeInsets.only(top: 8.h, right: 2.w),
              child: _buildSortingButton(context),
            ),
            Padding(
              padding: EdgeInsets.only(top: 8.h, right: 10.w),
              child: PopupMenuButton<String>(
                icon: Icon(
                  Icons.more_vert,
                  size: 10.sp, // 기존보다 살짝 큰 크기
                  color: const Color(0xFF646464),
                ),
                color: const Color(0xFFfefefe),
                offset: Offset(-20.h, 20.w),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(12),
                ),
                constraints: BoxConstraints(
                  minWidth: 70.w,
                  maxWidth: 70.w,
                ),
                onSelected: (value) {
                  if (value == '선택') {
                    _toggleSelectionMode();
                  } else if (value == '검색') {
                    _toggleSearchMode();
                  }
                },
                itemBuilder: (context) => [
                  PopupMenuItem(
                    value: '선택',
                    height: 45.h,
                    child: Padding(
                      padding: EdgeInsets.only(bottom: 8.h),
                      child: Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        crossAxisAlignment: CrossAxisAlignment.center,
                        children: [
                          Text('선택',
                              style: TextStyle(
                                fontSize: 5.sp,
                                fontWeight: FontWeight.w600,
                                color: const Color(0xFF646464),
                              )),
                          Icon(
                            Icons.check_circle_outline,
                            color: Color(0xFF646464),
                          ),
                        ],
                      ),
                    ),
                  ),
                  PopupMenuItem(
                    enabled: false,
                    height: 1.h,
                    padding: EdgeInsets.only(bottom: 8.h),
                    child: Divider(
                      height: 1,
                      thickness: 1,
                      color: Color(0xFFEEEEEE),
                    ),
                  ),
                  PopupMenuItem(
                    value: '검색',
                    height: 40.h,
                    child: Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      crossAxisAlignment: CrossAxisAlignment.center,
                      children: [
                        Text('검색',
                            style: TextStyle(
                              fontSize: 5.sp,
                              fontWeight: FontWeight.w600,
                              color: const Color(0xFF646464),
                            )),
                        Icon(
                          Icons.search,
                          color: Color(0xFF646464),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
          ],
        ],
      ),
      body: Padding(
        padding: EdgeInsets.symmetric(horizontal: 25.w, vertical: 30.h),
        child: Column(
          children: [
            Expanded(
              child: GridView.count(
                crossAxisCount: 3,
                childAspectRatio: 0.64.h,
                crossAxisSpacing: 35.w,
                mainAxisSpacing: 15.h,
                children: [
                  if (!_isSearchMode)
                    GestureDetector(
                      onTap: () {
                        showDialog(
                          context: context,
                          builder: (_) => AddSheetDialog(
                            onSubmit: (sheetName, artistName, filePath) async {
                              try {
                                final newSheet = await addSheet(
                                    sheetName, artistName, filePath!);
                                newSheet.artistName = artistName;
                                setState(() {
                                  _sheets.add(newSheet);
                                });
                              } catch (e) {
                                // 오류 처리
                                // ignore: use_build_context_synchronously
                                ScaffoldMessenger.of(context).showSnackBar(
                                    SnackBar(content: Text('Error: $e')));
                                print(e);
                              }
                            },
                          ),
                        );
                      },
                      child: Column(
                        children: [
                          AspectRatio(
                            aspectRatio: 0.9.h,
                            child: Container(
                              decoration: BoxDecoration(
                                color: Colors.white,
                                borderRadius: BorderRadius.circular(12),
                                boxShadow: const [
                                  BoxShadow(
                                    color: const Color(0xFFd9d9d9),
                                    blurRadius: 4,
                                    offset: Offset(0, 4),
                                  ),
                                ],
                              ),
                              child: Center(
                                child: Icon(
                                  Icons.add,
                                  size: 13.sp,
                                  color: Color(0xffD97D6C),
                                ),
                              ),
                            ),
                          ),
                          SizedBox(height: 15.h),
                          Text(
                            '악보 추가',
                            style: TextStyle(
                                fontSize: 6.sp,
                                fontWeight: FontWeight.w800,
                                color: const Color(0xFF646464)),
                          ),
                        ],
                      ),
                    ),
                  ...displayedSheets.map((sheet) {
                    final iconColor = _iconColorMap[sheet.color] ??
                        Colors.black.withOpacity(0.4);

                    return GestureDetector(
                      onTap: () {
                        if (_isSelectionMode) {
                          _onSheetSelected(sheet);
                        } else {
                          _confirmPracticeStart(sheet);
                        }
                      },
                      child: Stack(
                        children: [
                          SheetCard(sheet: sheet, iconColor: iconColor),
                          if (_isSelectionMode)
                            Positioned(
                              bottom: 105.h,
                              left: 0,
                              right: 0,
                              child: GestureDetector(
                                onTap: () => _onSheetSelected(sheet),
                                child: Container(
                                  width: 35.w,
                                  height: 35.h,
                                  decoration: BoxDecoration(
                                    shape: BoxShape.circle,
                                    border: Border.all(
                                      color: sheet.isSelected
                                          ? Colors.transparent
                                          : const Color(0xFF646464),
                                      width: 2,
                                      style: sheet.isSelected
                                          ? BorderStyle.none
                                          : BorderStyle.solid,
                                    ),
                                    color: sheet.isSelected
                                        ? const Color(0xFF646464)
                                        : Colors.transparent,
                                  ),
                                  child: sheet.isSelected
                                      ? Icon(Icons.check,
                                          color: Colors.white, size: 8.sp)
                                      : null,
                                ),
                              ),
                            ),
                        ],
                      ),
                    );
                  }).toList(),
                ],
              ),
            ),
            if (_isSelectionMode) ...[
              if (selectedSheets.length == 1)
                Container(
                  height: MediaQuery.of(context).size.height * 0.12,
                  padding: EdgeInsets.symmetric(vertical: 3.h),
                  decoration: BoxDecoration(
                    borderRadius: BorderRadius.circular(32),
                    boxShadow: [
                      BoxShadow(
                        color: const Color(0xFFd9d9d9),
                        blurRadius: 4,
                        offset: const Offset(0, 2),
                      ),
                    ],
                    color: Colors.white,
                  ),
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.spaceAround,
                    children: [
                      TextButton(
                        onPressed: () => _renameSheet(selectedSheets.first),
                        style: TextButton.styleFrom(
                            textStyle: TextStyle(
                          fontSize: 6.sp,
                        )),
                        child: const Text('이름 변경',
                            style: TextStyle(
                                fontWeight: FontWeight.bold,
                                color: Color(0xFF646464))),
                      ),
                      TextButton(
                        onPressed: () =>
                            _changeSheetColor(selectedSheets.first),
                        style: TextButton.styleFrom(
                            textStyle: TextStyle(
                          fontSize: 6.sp,
                        )),
                        child: const Text('색상 변경',
                            style: TextStyle(
                                fontWeight: FontWeight.bold,
                                color: Color(0xFF646464))),
                      ),
                      TextButton(
                        onPressed: () => _confirmDeleteMultiple(selectedSheets),
                        style: TextButton.styleFrom(
                            textStyle: TextStyle(
                          fontSize: 6.sp,
                        )),
                        child: const Text('삭제',
                            style: TextStyle(
                                fontWeight: FontWeight.bold,
                                color: Color(0xFFd97d6c))),
                      ),
                    ],
                  ),
                )
              else if (selectedSheets.length > 1)
                Container(
                  height: MediaQuery.of(context).size.height * 0.12,
                  padding: EdgeInsets.symmetric(vertical: 3.h),
                  decoration: BoxDecoration(
                    borderRadius: BorderRadius.circular(32),
                    boxShadow: [
                      BoxShadow(
                        color: const Color(0xFFd9d9d9),
                        blurRadius: 4,
                        offset: const Offset(0, 2),
                      ),
                    ],
                    color: Colors.white,
                  ),
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.spaceAround,
                    children: [
                      TextButton(
                        style: TextButton.styleFrom(
                            textStyle: TextStyle(
                          fontSize: 6.sp,
                        )),
                        onPressed: () => _confirmDeleteMultiple(selectedSheets),
                        child: const Text('삭제',
                            style: TextStyle(
                                fontWeight: FontWeight.bold,
                                color: Color(0xFFd97d6c))),
                      ),
                    ],
                  ),
                )
            ],
          ],
        ),
      ),
    );
  }

  Widget _buildSortingButton(BuildContext context) {
    return InkWell(
      onTap: () {
        showModalBottomSheet(
          context: context,
          backgroundColor: Colors.white,
          barrierColor: Colors.transparent,
          shape: const RoundedRectangleBorder(
            borderRadius: BorderRadius.vertical(top: Radius.circular(16)),
          ),
          builder: (context) => SizedBox(
            width: MediaQuery.of(context).size.width * 0.5,
            child: Padding(
              padding: EdgeInsets.only(bottom: 20.h),
              child: Wrap(
                children: [
                  Padding(
                    padding: EdgeInsets.symmetric(vertical: 12.h),
                    child: Center(
                      child: Text(
                        '정렬',
                        style: TextStyle(
                            fontSize: 6.sp,
                            color: const Color(0xff646464),
                            fontWeight: FontWeight.bold),
                      ),
                    ),
                  ),
                  _buildSortTile(SortOption.date, '생성 날짜 순'),
                  _buildSortTile(SortOption.name, '이름 순'),
                  _buildSortTile(SortOption.recentPractice, '최근 연습한 순'),
                ],
              ),
            ),
          ),
        );
      },
      child: Container(
        padding: EdgeInsets.symmetric(horizontal: 6.w, vertical: 10.h),
        decoration: BoxDecoration(
            color: Colors.white,
            borderRadius: BorderRadius.circular(32),
            border: Border.all(color: const Color(0xFFDFDFDF), width: 2)),
        child: Row(
          children: [
            Icon(
              Icons.swap_vert,
              size: 9.sp, // 아이콘 크기 설정
              color: const Color(0xFF646464),
            ),
            SizedBox(width: 3.w),
            Text(
              _sortLabel,
              style: TextStyle(
                  fontSize: 5.5.sp,
                  fontWeight: FontWeight.bold,
                  color: const Color(0xFF646464)),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildSortTile(SortOption option, String label) {
    final isSelected = _selectedSort == option;

    return ListTile(
      contentPadding: EdgeInsets.symmetric(horizontal: 10.w),
      title: Text(
        label,
        style: TextStyle(
          fontSize: 6.sp,
          fontWeight: FontWeight.w600,
          color: isSelected ? Color(0xffd97d6c) : Color(0xff646464),
        ),
      ),
      trailing: isSelected
          ? Icon(Icons.check, color: Color(0xffd97d6c), size: 9.sp)
          : null,
      onTap: () {
        _onSortSelected(option);
        Navigator.pop(context);
      },
    );
  }
}
