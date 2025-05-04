import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

import '../../models/sheet.dart';
import 'widgets/sheet_card.dart';
import 'widgets/add_sheet_dialog.dart';

void main() => runApp(const DrumSheetScreen());

class DrumSheetScreen extends StatelessWidget {
  const DrumSheetScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
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

  // 악보 리스트 샘플 데이터
  final List<Sheet> _sheets = [
    Sheet(
        title: '가',
        createdDate: DateTime(2023, 1, 5),
        lastPracticedDate: DateTime(2024, 3, 21),
        color: Color(0xFFBEBEBE)),
    Sheet(
        title: '나',
        createdDate: DateTime(2024, 10, 12),
        lastPracticedDate: DateTime(2024, 12, 2),
        color: Color(0xFFF4B3B3)),
    Sheet(
        title: '다',
        createdDate: DateTime(2028, 8, 30),
        lastPracticedDate: DateTime(2030, 1, 15),
        color: Color(0xFFF4DDB3)),
    Sheet(
      title: '라',
      createdDate: DateTime(2019, 8, 30),
      lastPracticedDate: DateTime(2020, 1, 15),
      color: Color(0xFFb3f4b5),
    ),
    Sheet(
      title: '마',
      createdDate: DateTime(2021, 8, 30),
      lastPracticedDate: DateTime(2024, 1, 15),
      color: Color(0xFFb3eaf4),
    ),
    Sheet(
      title: '바',
      createdDate: DateTime(2020, 4, 30),
      lastPracticedDate: DateTime(2025, 1, 15),
      color: Color(0xFFdcb3f4),
    ),
    Sheet(
        title: '사',
        createdDate: DateTime(2001, 8, 30),
        lastPracticedDate: DateTime(2025, 1, 15)),
  ];

  // API 관련 메서드들
  Future<List<Sheet>> fetchSheets() async {
    final response = await http.get(Uri.parse('http://10.0.2.2:28080/sheets'));
    if (response.statusCode == 200) {
      List jsonResponse = json.decode(response.body);
      return jsonResponse.map((sheet) => Sheet.fromJson(sheet)).toList();
    } else {
      throw Exception('악보 로딩에 실패했습니다');
    }
  }

  Future<Sheet> addSheet(String title, String artist) async {
    final response = await http.post(
      Uri.parse('http://10.0.2.2:28080/sheets'),
      headers: <String, String>{
        'Content-Type': 'application/json; charset=UTF-8',
      },
      body: jsonEncode(<String, String>{
        'title': title,
        'artist': artist,
      }),
    );

    if (response.statusCode == 201) {
      return Sheet.fromJson(json.decode(response.body));
    } else {
      throw Exception('Failed to add sheet');
    }
  }

  Future<void> updateSheet(int id, String title) async {
    final response = await http.put(
      Uri.parse('http://10.0.2.2:28080/sheets/$id'),
      headers: <String, String>{
        'Content-Type': 'application/json; charset=UTF-8',
      },
      body: jsonEncode(<String, String>{'title': title}),
    );

    if (response.statusCode != 200) {
      throw Exception('Failed to update sheet');
    }
  }

  Future<void> deleteSheet(int id) async {
    final response = await http.delete(
      Uri.parse('http://10.0.2.2:28080/sheets/$id'),
      headers: <String, String>{
        'Content-Type': 'application/json; charset=UTF-8',
      },
    );

    if (response.statusCode != 200) {
      throw Exception('악보 삭제');
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
        return '날짜순';
      case SortOption.name:
        return '이름순';
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
              onPressed: () {
                setState(() {
                  sheet.title = controller.text;
                });
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

  void _changeSheetColor(Sheet sheet) {
    showDialog(
      context: context,
      barrierColor: Colors.transparent,
      builder: (context) {
        return Stack(
          children: [
            Positioned(
              bottom: 60,
              left: MediaQuery.of(context).size.width / 2 -
                  (customColors.length * 24) / 2 +
                  100,
              child: Material(
                color: Colors.transparent,
                child: Container(
                  padding:
                      const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
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
                        onTap: () {
                          setState(() {
                            sheet.color = color;
                          });
                          Navigator.of(context).pop();
                        },
                        child: Container(
                          width: 28,
                          height: 28,
                          margin: const EdgeInsets.symmetric(horizontal: 6),
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

  void _confirmDeleteMultiple(List<Sheet> sheets) {
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
            width: 300,
            child: Padding(
              padding: const EdgeInsets.symmetric(vertical: 24, horizontal: 16),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  const Text(
                    '선택한 악보를 삭제하시겠습니까?',
                    style: TextStyle(
                      fontSize: 16,
                      fontWeight: FontWeight.w600,
                      color: Color(0xFF646464),
                    ),
                  ),
                  const SizedBox(height: 24),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                    children: [
                      ElevatedButton(
                        style: ElevatedButton.styleFrom(
                          backgroundColor: const Color(0xFFF2F2F2),
                          minimumSize: const Size(100, 45),
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
                          minimumSize: const Size(100, 45),
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(12),
                          ),
                        ),
                        onPressed: () {
                          setState(() {
                            _sheets
                                .removeWhere((sheet) => sheets.contains(sheet));
                          });
                          Navigator.of(context).pop();
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
            width: 300,
            child: Padding(
              padding: const EdgeInsets.symmetric(vertical: 24, horizontal: 16),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Text(
                    '${sheet.title}-가수',
                    style: const TextStyle(
                      fontSize: 16,
                      fontWeight: FontWeight.w600,
                      color: Color(0xFF646464),
                    ),
                  ),
                  const SizedBox(height: 12),
                  const Text(
                    '연주를 시작하시겠습니까?',
                    style: TextStyle(
                      fontSize: 13,
                      color: Color(0xFF646464),
                    ),
                  ),
                  const SizedBox(height: 24),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                    children: [
                      ElevatedButton(
                        style: ElevatedButton.styleFrom(
                          backgroundColor: const Color(0xFFF2F2F2),
                          minimumSize: const Size(100, 45),
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
                          minimumSize: const Size(100, 45),
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(12),
                          ),
                        ),
                        onPressed: () {
                          // 연주 시작 페이지로 이동하는 코드 추가하기
                          Navigator.of(context).pop();
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
        title: Row(
          children: [
            Icon(Icons.library_music, color: Colors.black54),
            SizedBox(width: 8),
            Text(
              '악보 목록',
              style: TextStyle(color: Colors.black87),
            ),
          ],
        ),
        backgroundColor: Color(0xFFF5F5F5),
        elevation: 0,
        toolbarHeight: 50,
        actions: [
          if (_isSelectionMode) ...[
            Container(
              margin: const EdgeInsets.only(right: 8),
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(32),
                boxShadow: [
                  BoxShadow(
                    color: Colors.black.withValues(alpha: 0.1),
                    blurRadius: 4,
                    offset: const Offset(0, 2),
                  ),
                ],
              ),
              child: TextButton(
                onPressed: _toggleSelectAll,
                style: TextButton.styleFrom(
                  padding:
                      const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(32),
                  ),
                ),
                child: Text(
                  _isAllSelected ? '전체 선택 해제' : '전체 선택',
                  style: const TextStyle(
                    fontSize: 14,
                    fontWeight: FontWeight.w600,
                    color: Color(0xFF595959),
                  ),
                ),
              ),
            ),
            Container(
              margin: const EdgeInsets.only(right: 8),
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(32),
                boxShadow: [
                  BoxShadow(
                    color: Colors.black.withValues(alpha: 0.1),
                    blurRadius: 4,
                    offset: const Offset(0, 2),
                  ),
                ],
              ),
              child: TextButton(
                onPressed: _toggleSelectionMode,
                style: TextButton.styleFrom(
                  padding:
                      const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(32),
                  ),
                ),
                child: const Text(
                  '완료',
                  style: TextStyle(
                    fontSize: 14,
                    fontWeight: FontWeight.w600,
                    color: Color(0xFF595959),
                  ),
                ),
              ),
            ),
          ] else if (_isSearchMode) ...[
            Container(
              width: 250,
              height: 32,
              margin: const EdgeInsets.symmetric(vertical: 8, horizontal: 8),
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
                  const Padding(
                    padding: EdgeInsets.only(left: 12),
                    child: Icon(
                      Icons.search,
                      size: 18,
                      color: Color(0xFF595959),
                    ),
                  ),
                  Expanded(
                    child: TextField(
                      controller: _searchController,
                      autofocus: true,
                      decoration: const InputDecoration(
                        hintText: '검색어를 입력해주세요',
                        hintStyle: TextStyle(
                          color: Color(0xFF949494),
                          fontSize: 13,
                        ),
                        border: InputBorder.none,
                        contentPadding:
                            EdgeInsets.symmetric(horizontal: 8, vertical: 8),
                      ),
                      style: const TextStyle(
                        color: Color(0xFF595959),
                        fontSize: 13,
                      ),
                      onChanged: _performSearch,
                    ),
                  ),
                  IconButton(
                    padding: const EdgeInsets.all(4),
                    constraints: const BoxConstraints(),
                    icon: const Icon(
                      Icons.close,
                      size: 18,
                      color: Color(0xFF595959),
                    ),
                    onPressed: _toggleSearchMode,
                  ),
                  const SizedBox(width: 8),
                ],
              ),
            ),
          ] else ...[
            _buildSortingButton(context),
            PopupMenuButton<String>(
              color: const Color(0xFFfefefe),
              offset: const Offset(-20, 50),
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(12),
              ),
              constraints: const BoxConstraints(
                minWidth: 160,
                maxWidth: 160,
              ),
              onSelected: (value) {
                if (value == '선택') {
                  _toggleSelectionMode();
                } else if (value == '검색') {
                  _toggleSearchMode();
                }
              },
              itemBuilder: (context) => [
                const PopupMenuItem(
                  value: '선택',
                  height: 40,
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Text('선택'),
                      Icon(
                        Icons.check_circle_outline,
                        color: Color(0xFF595959),
                      ),
                    ],
                  ),
                ),
                const PopupMenuItem(
                  enabled: false,
                  height: 1,
                  padding: EdgeInsets.zero,
                  child: Divider(
                    height: 1,
                    thickness: 1,
                    color: Color(0xFFEEEEEE),
                  ),
                ),
                const PopupMenuItem(
                  value: '검색',
                  height: 40,
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Text('검색'),
                      Icon(
                        Icons.search,
                        color: Color(0xFF595959),
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ],
        ],
      ),
      body: Padding(
        padding: const EdgeInsets.all(10),
        child: Column(
          children: [
            Expanded(
              child: GridView.count(
                crossAxisCount: 3,
                childAspectRatio: 0.6,
                crossAxisSpacing: 16,
                mainAxisSpacing: 16,
                children: [
                  if (!_isSearchMode)
                    GestureDetector(
                      onTap: () {
                        showDialog(
                          context: context,
                          builder: (_) => AddSheetDialog(
                            onSubmit: (sheetName, artistName) async {
                              try {
                                final newSheet =
                                    await addSheet(sheetName, artistName);
                                setState(() {
                                  _sheets.add(newSheet);
                                });
                              } catch (e) {
                                // 오류 처리
                                // ignore: use_build_context_synchronously
                                ScaffoldMessenger.of(context).showSnackBar(
                                    SnackBar(content: Text('Error: $e')));
                              }
                            },
                          ),
                        );
                      },
                      child: Column(
                        children: [
                          Expanded(
                            child: Container(
                              decoration: BoxDecoration(
                                color: Colors.white,
                                borderRadius: BorderRadius.circular(12),
                                boxShadow: const [
                                  BoxShadow(
                                    color: Colors.black12,
                                    blurRadius: 4,
                                    offset: Offset(0, 2),
                                  ),
                                ],
                              ),
                              child: const Center(
                                child: Icon(
                                  Icons.add,
                                  size: 36,
                                  color: Colors.redAccent,
                                ),
                              ),
                            ),
                          ),
                          const SizedBox(height: 25),
                          const Text(
                            '악보 추가',
                            style:
                                TextStyle(fontSize: 14, color: Colors.black87),
                          ),
                        ],
                      ),
                    ),
                  ...displayedSheets.map((sheet) => GestureDetector(
                        onTap: () {
                          if (_isSelectionMode) {
                            _onSheetSelected(sheet);
                          } else {
                            _confirmPracticeStart(sheet);
                          }
                        },
                        child: SizedBox(
                          width: 100,
                          height: 150,
                          child: Stack(
                            children: [
                              SheetCard(sheet: sheet),
                              if (_isSelectionMode)
                                Positioned(
                                  bottom: 65,
                                  left: 0,
                                  right: 0,
                                  child: GestureDetector(
                                    onTap: () => _onSheetSelected(sheet),
                                    child: Container(
                                      width: 28,
                                      height: 28,
                                      decoration: BoxDecoration(
                                        shape: BoxShape.circle,
                                        border: Border.all(
                                          color: sheet.isSelected
                                              ? Colors.transparent
                                              : Colors.black26,
                                          width: 2,
                                          style: sheet.isSelected
                                              ? BorderStyle.none
                                              : BorderStyle.solid,
                                        ),
                                        color: sheet.isSelected
                                            ? Colors.black54
                                            : Colors.transparent,
                                      ),
                                      child: sheet.isSelected
                                          ? const Icon(Icons.check,
                                              color: Colors.white, size: 18)
                                          : null,
                                    ),
                                  ),
                                ),
                            ],
                          ),
                        ),
                      )),
                ],
              ),
            ),
            if (_isSelectionMode) ...[
              if (selectedSheets.length == 1)
                Container(
                  height: MediaQuery.of(context).size.height * 0.12,
                  padding: const EdgeInsets.symmetric(vertical: 3),
                  decoration: BoxDecoration(
                    borderRadius: BorderRadius.circular(32),
                    color: Colors.white,
                  ),
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.spaceAround,
                    children: [
                      TextButton(
                        onPressed: () => _renameSheet(selectedSheets.first),
                        child: const Text('이름 변경',
                            style: TextStyle(
                                fontWeight: FontWeight.bold,
                                color: Color(0xFF646464))),
                      ),
                      TextButton(
                        onPressed: () =>
                            _changeSheetColor(selectedSheets.first),
                        child: const Text('색상 변경',
                            style: TextStyle(
                                fontWeight: FontWeight.bold,
                                color: Color(0xFF646464))),
                      ),
                      TextButton(
                        onPressed: () => _confirmDeleteMultiple(selectedSheets),
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
                  padding: const EdgeInsets.symmetric(vertical: 3),
                  decoration: BoxDecoration(
                    borderRadius: BorderRadius.circular(32),
                    color: Colors.white,
                  ),
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.spaceAround,
                    children: [
                      TextButton(
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
              padding: const EdgeInsets.only(bottom: 20),
              child: Wrap(
                children: [
                  const Padding(
                    padding: EdgeInsets.symmetric(vertical: 12.0),
                    child: Center(
                      child: Text(
                        '정렬',
                        style: TextStyle(
                            fontSize: 18, fontWeight: FontWeight.bold),
                      ),
                    ),
                  ),
                  _buildSortTile(SortOption.date, '날짜순'),
                  _buildSortTile(SortOption.name, '이름순'),
                  _buildSortTile(SortOption.recentPractice, '최근 연습한 순'),
                ],
              ),
            ),
          ),
        );
      },
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 13, vertical: 8),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(32),
          boxShadow: const [
            BoxShadow(
              color: Colors.black12,
              blurRadius: 6,
              offset: Offset(0, 2),
            ),
          ],
        ),
        child: Row(
          children: [
            Icon(
              Icons.swap_vert,
              color: const Color.fromARGB(119, 0, 0, 0),
            ),
            const SizedBox(width: 8),
            Text(
              _sortLabel,
              style: const TextStyle(fontSize: 14, color: Colors.black),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildSortTile(SortOption option, String label) {
    final isSelected = _selectedSort == option;

    return ListTile(
      title: Text(
        label,
        style: TextStyle(
          fontSize: 16,
          fontWeight: FontWeight.w600,
          color: isSelected ? Color(0xffd97d6c) : Color(0xff646464),
        ),
      ),
      trailing: isSelected
          ? const Icon(Icons.check, color: Color(0xffd97d6c), size: 20)
          : null,
      onTap: () {
        _onSortSelected(option);
        Navigator.pop(context);
      },
    );
  }
}
