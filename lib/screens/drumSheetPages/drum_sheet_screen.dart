import 'package:flutter/material.dart';

void main() => runApp(const DrumSheetScreen());

class DrumSheetScreen extends StatelessWidget {
  const DrumSheetScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: '악보 목록', // 페이지 제목
      theme: ThemeData(
        fontFamily: 'Pretendard', // 커스텀 폰트 사용
        scaffoldBackgroundColor: const Color(0xFFf5f5f5), // 앱 배경 색상
      ),
      home: const SheetListScreen(), // 메인 홈 화면
    );
  }
}

// 정렬 기준 옵션 enum
enum SortOption { date, name, recentPractice }

// 악보 데이터 모델
class Sheet {
  String title; // 악보 이름
  final DateTime createdDate; // 생성 날짜
  final DateTime lastPracticedDate; // 마지막 연습 날짜
  Color color; // 카드 배경 색상
  bool isSelected; // 선택 여부

  // 악보 데이터 모델 생성자
  Sheet({
    required this.title,
    required this.createdDate,
    required this.lastPracticedDate,
    this.color = const Color(0xFFBEBEBE), // 악보 카드 기본 배경 색상은 회색
    this.isSelected = false, // 기본값은 선택되지 않음
  });
}

// 악보 리스트 화면 StatefulWidget
class SheetListScreen extends StatefulWidget {
  const SheetListScreen({super.key});

  @override
  State<SheetListScreen> createState() => _SheetListScreenState();
}

class _SheetListScreenState extends State<SheetListScreen> {
  SortOption _selectedSort = SortOption.date; // 현재 정렬 기준 (기본값은 날짜순)
  bool _isSelectionMode = false; // 선택 모드 상태 (기본값은 선택모드 아님)
  // 이 값이 true가 되면 악보 선택 가능 상태로 전환됨
  bool _isAllSelected = false; // 모든 아이템 선택 여부

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

  // 현재 정렬 기준에 따라 정렬된 리스트 반환
  List<Sheet> get _sortedSheets {
    List<Sheet> sorted =
        List.from(_sheets); // 리스트 복사. 원본 리스트 변경하지 않고 정렬된 결과를 새로 만들기 위해 사용.
    switch (_selectedSort) {
      case SortOption.date:
        sorted
            .sort((a, b) => b.createdDate.compareTo(a.createdDate)); // 날짜 순 정렬
        break;
      case SortOption.name:
        sorted.sort((a, b) => a.title.compareTo(b.title)); // 이름 순 정렬
        break;
      case SortOption.recentPractice:
        sorted.sort((a, b) =>
            b.lastPracticedDate.compareTo(a.lastPracticedDate)); // 최근 연습 순 정렬
        break;
    }
    return sorted;
  }

  // 현재 정렬 기준 라벨 반환
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

  // 정렬 기준 선택 시 실행
  void _onSortSelected(SortOption option) {
    setState(() {
      _selectedSort = option; // 정렬 기준 변경
    });
  }

  // 선택 모드 ON/OFF 토글
  void _toggleSelectionMode() {
    setState(() {
      _isSelectionMode = !_isSelectionMode; // 선택 모드 상태 바꾸기
      if (!_isSelectionMode) {
        // 선택 모드 종료 시 모든 선택 해제
        for (var sheet in _sheets) {
          sheet.isSelected = false;
        }
      }
    });
  }

  // 전체 악보 선택/해제
  void _toggleSelectAll() {
    setState(() {
      _isAllSelected = !_isAllSelected;
      for (var sheet in _sheets) {
        sheet.isSelected = _isAllSelected; // 모든 악보의 선택 상태를 변경
      }
    });
  }

  // 개별 악보 선택/해제
  void _onSheetSelected(Sheet sheet) {
    setState(() {
      sheet.isSelected = !sheet.isSelected; // 선택 상태 반전
    });
  }

  // 이름 변경 로직 자리
  void _renameSheet(Sheet sheet) {
    final TextEditingController controller =
        TextEditingController(text: sheet.title); // 기존의 악보 파일 이름을 미리 텍스트 필드에 표시

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
              onPressed: () {
                Navigator.of(context).pop(); // 취소 버튼
              },
              child: const Text('취소'),
            ),
            ElevatedButton(
              onPressed: () {
                setState(() {
                  sheet.title = controller.text; // 이름 변경
                });
                Navigator.of(context).pop(); // 저장 후 다이얼로그 닫기
              },
              child: const Text('저장'),
            ),
          ],
        );
      },
    );
  }

  // 악보 파일 색상 리스트 정의
  final List<Color> customColors = [
    Color(0xFFBEBEBE),
    Color(0xFFF4B3B3),
    Color(0xFFF4DDB3),
    Color(0xFFb3f4b5),
    Color(0xFFb3eaf4),
    Color(0xFFdcb3f4),
  ];

  // 색상 변경 함수
  void _changeSheetColor(Sheet sheet) {
    final RenderBox button = context.findRenderObject() as RenderBox;
    final Offset position = button.localToGlobal(Offset.zero); // 색상 선택 버튼 위치

    showDialog(
      context: context,
      barrierColor: Colors.transparent,
      builder: (context) {
        return Stack(
          children: [
            Positioned(
              bottom: 60, // 하단 작업 메뉴 위로 색상 팔레트 위치
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
                            sheet.color = color; // 색상 변경
                          });
                          Navigator.of(context).pop(); // 다이얼로그 닫기
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

  // 삭제 함수
  void _deleteSelectedSheets() {
    setState(() {
      _sheets.removeWhere((sheet) => sheet.isSelected);
    });
  }

  // 악보 삭제
  void _deleteSheet(Sheet sheet) {
    setState(() {
      _sheets.remove(sheet); // 리스트에서 삭제
    });
  }

  void _confirmDelete(Sheet sheet) {
    showDialog(
      context: context,
      builder: (context) {
        return AlertDialog(
          iconColor: Color(0xFFf5f5f5),
          title: const Text('삭제하시겠습니까?'),
          actions: [
            TextButton(
              onPressed: () => Navigator.of(context).pop(),
              child: const Text(
                '아니오',
                style: TextStyle(
                  color: Color(0xFF646464),
                ),
              ),
            ),
            ElevatedButton(
              onPressed: () {
                _confirmDelete(sheet);
                Navigator.of(context).pop();
              },
              style: ElevatedButton.styleFrom(
                backgroundColor: Color(0xFFd97d6c),
              ),
              child: const Text(
                '예',
                style: TextStyle(
                  color: Color(0xFF646464),
                ),
              ),
            ),
          ],
        );
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    final selectedSheets =
        _sheets.where((sheet) => sheet.isSelected).toList(); // 선택된 악보들

    return Scaffold(
      appBar: AppBar(
        title: Row(
          children: [
            Icon(Icons.library_music, color: Colors.black54), // ← 추가된 아이콘
            SizedBox(width: 8),
            Text(
              '악보 목록',
              style: TextStyle(color: Colors.black87),
            ),
          ],
        ),
        backgroundColor: Color(0xFFF5F5F5),
        elevation: 0, // 그림자 없애기
        // primary: false, // 스크롤 시 배경색 변화 없게 하기. 이상함 이 코드 동작 안함.
        toolbarHeight: 50, // AppBar의 높이 설정
        actions: [
          // 정렬 아이콘 클릭 시 BottomSheet
          _buildSortingButton(context),
          // 우측 점3개 아이콘 (PopupMenu)
          PopupMenuButton<String>(
            color: const Color(0xFFfefefe),
            onSelected: (value) {
              if (value == '선택') {
                _toggleSelectionMode(); // 선택 모드 전환
              }
            },
            itemBuilder: (context) => [
              const PopupMenuItem(
                value: '선택',
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
                value: '검색',
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
      ),
      // 악보 보여주는 영역
      body: Padding(
        padding: const EdgeInsets.all(10),
        child: Column(
          children: [
            // 악보 리스트 (Grid 형태)
            Expanded(
              child: GridView.count(
                crossAxisCount: 3, // 한 줄에 3개의 악보
                childAspectRatio: 0.6, // 악보 비율
                crossAxisSpacing: 16,
                mainAxisSpacing: 16,
                children: [
                  AddSheetCard(), // 추가 버튼 카드
                  ..._sortedSheets.map((sheet) => GestureDetector(
                        onTap: () {
                          if (_isSelectionMode) {
                            _onSheetSelected(sheet); // 선택 모드일 경우 악보 선택
                          } else {
                            // 일반 모드에서의 동작 구현
                          }
                        },
                        child: Stack(
                          children: [
                            SheetCard(sheet: sheet), // 악보 카드
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
                      )),
                ],
              ),
            ),
            // 하단 작업 메뉴 (카드 1개만 선택 시 표시)
            if (selectedSheets.length == 1)
              Container(
                // color: Colors.white,
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
                      onPressed: () =>
                          _renameSheet(selectedSheets.first), // 이름 변경
                      child: const Text('이름 변경',
                          style: TextStyle(
                              fontWeight: FontWeight.bold,
                              color: Color(0xFF646464))),
                    ),
                    TextButton(
                      onPressed: () =>
                          _changeSheetColor(selectedSheets.first), // 색상 변경
                      child: const Text('색상 변경',
                          style: TextStyle(
                              fontWeight: FontWeight.bold,
                              color: Color(0xFF646464))),
                    ),
                    TextButton(
                      onPressed: () =>
                          _confirmDelete(selectedSheets.first), // 삭제
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
                // color: Colors.white,
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
                      onPressed: () =>
                          _confirmDelete(selectedSheets.first), // 삭제
                      child: const Text('삭제',
                          style: TextStyle(
                              fontWeight: FontWeight.bold,
                              color: Color(0xFFd97d6c))),
                    ),
                  ],
                ),
              )
          ],
        ),
      ),
    );
  }

  // 정렬 버튼
  Widget _buildSortingButton(BuildContext context) {
    return InkWell(
      onTap: () {
        showModalBottomSheet(
          context: context,
          backgroundColor: Colors.white,
          barrierColor: Colors.transparent, // 배경 어두워지지 않게 설정
          shape: const RoundedRectangleBorder(
            borderRadius: BorderRadius.vertical(top: Radius.circular(16)),
          ),
          builder: (context) => SizedBox(
            width: MediaQuery.of(context).size.width * 0.5, // 화면 너비의 50%로 설정
            child: Padding(
              padding: const EdgeInsets.only(bottom: 20), // 하단 padding
              child: Wrap(
                //   mainAxisSize: MainAxisSize.min,
                children: [
                  const Padding(
                    //   padding: EdgeInsets.only(bottom: 28.0),
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
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
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
              Icons.swap_vert, // 정렬 아이콘
              color: Colors.grey,
            ),
            const SizedBox(width: 8),
            Text(
              _sortLabel, // 현재 선택된 정렬 기준 텍스트
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

// /추가 카드 UI (플러스 버튼)
class AddSheetCard extends StatelessWidget {
  const AddSheetCard({super.key});

  @override
  Widget build(BuildContext context) {
    return Column(
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
                color: Colors.redAccent, // 추가 버튼 아이콘 색상
              ),
            ),
          ),
        ),
        const SizedBox(height: 25),
        const Text(
          '악보 추가', // 버튼 하단 텍스트
          style: TextStyle(fontSize: 14, color: Colors.black87),
        ),
      ],
    );
  }
}

// 악보 카드 UI
class SheetCard extends StatelessWidget {
  final Sheet sheet;

  const SheetCard({super.key, required this.sheet});

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Expanded(
          child: Container(
            decoration: BoxDecoration(
              color: sheet.color, // 악보 카드 색상
              borderRadius: BorderRadius.circular(12),
              boxShadow: const [
                BoxShadow(
                  color: Colors.black26,
                  blurRadius: 12,
                  offset: Offset(0, 6),
                ),
              ],
            ),
            child: Column(
              children: [
                Expanded(
                  child: Center(
                    child: Icon(
                      Icons.music_note, // 악보 아이콘
                      color: Colors.black.withValues(alpha: 0.4),
                      size: 36,
                    ), // 악보 아이콘
                  ),
                ),
              ],
            ),
          ),
        ),
        const SizedBox(height: 8),
        Text(sheet.title, style: const TextStyle(fontSize: 14)), // 악보 이름
        Text(
          '${sheet.createdDate.year}.${sheet.createdDate.month.toString().padLeft(2, '0')}.${sheet.createdDate.day.toString().padLeft(2, '0')}', // 생성일자 포맷
          style: const TextStyle(fontSize: 12, color: Colors.grey),
        ),
      ],
    );
  }
}
