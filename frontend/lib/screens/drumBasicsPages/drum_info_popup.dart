import 'package:flutter/material.dart';

/// 드럼 기초 팝업 창
class DrumInfoPopup extends StatefulWidget {
  final String title; // 팝업 제목
  final String? imagePath; // 이미지 경로
  final String description; // 설명 텍스트 추가

  const DrumInfoPopup({
    super.key,
    required this.title, // 팝업 제목 필수 입력
    required this.imagePath, // 이미지 경로를 선택적으로 받음
    required this.description, // 설명 텍스트 필수 입력
  });

  @override
  State<DrumInfoPopup> createState() => _DrumInfoPopupState();
}

class _DrumInfoPopupState extends State<DrumInfoPopup> {
  final ScrollController _scrollController = ScrollController(); // 스크롤 컨트롤러

  @override
  void dispose() {
    _scrollController.dispose(); // 위젯이 제거될 때 스크롤 컨트롤러 해제 (메모리 누수 방지)
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    // 현재 화면의 가로 및 세로 크기를 가져옴
    final double screenHeight = MediaQuery.of(context).size.height;
    final double screenWidth = MediaQuery.of(context).size.width;

    return Dialog(
      // 둥근 모서리를 가진 다이얼로그 창
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10.0)),
      backgroundColor: Colors.white, // 배경색 흰색 설정
      child: SizedBox(
        width: screenWidth * 0.8, // 팝업 창 너비: 화면의 80% 차지
        height: screenHeight, // 팝업 창 높이: 화면 전체 차지
        child: Stack(
          children: [
            Padding(
              padding: const EdgeInsets.fromLTRB(30, 20, 30, 20),
              child: Column(
                children: [
                  _buildTitle(), // 제목: 스크롤되지 않음
                  const SizedBox(height: 20), // 제목과 내용 사이 간격
                  Expanded(
                    child: Stack(
                      children: [
                        _buildScrollbarBackground(), // 스크롤바 배경 추가(그림자)
                        ScrollbarTheme(
                          data: _customScrollbarTheme(), // 스크롤바 테마 적용
                          child: Scrollbar(
                            controller: _scrollController, // 스크롤 컨트롤러 연결
                            trackVisibility: true, // 스크롤 트랙(스크롤이 움직이는 영역) 표시
                            thumbVisibility: true, // 스크롤바 표시
                            interactive: true, // 스크롤바 드래그 가능하게 설정
                            child: SingleChildScrollView(
                              controller: _scrollController, // 스크롤 컨트롤러 연결
                              physics:
                                  const AlwaysScrollableScrollPhysics(), // 항상 스크롤 가능하도록 설정
                              child: Column(
                                children: [
                                  _buildImage(), // 이미지 (옵션)
                                  const SizedBox(height: 20), // 이미지와 설명 사이 간격
                                  _buildDescription(), // 설명 텍스트
                                ],
                              ),
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
            _buildCloseButton(context), // 닫기 버튼(오른쪽 상단)
          ],
        ),
      ),
    );
  }

  /// 스크롤바 배경 그림자 효과 추가
  Positioned _buildScrollbarBackground() {
    // 스크롤바의 테두리를 설정하는 테마 없음.
    // 그 대신에, 스크롤바 배경 그림자 효과 추가
    return Positioned(
      right: 3, // 부모 위젯(Stack)의 오른쪽에서 3픽셀 떨어진 위치
      top: 0,
      bottom: 0,
      child: DecoratedBox(
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(10), // 둥근 테두리
          boxShadow: [
            BoxShadow(
              color: Colors.black.withValues(alpha: 0.3), // 그림자 색상(연한 검정)
              spreadRadius: 2, // 그림자 퍼지는 정도
              blurRadius: 2, // 흐림 효과
              offset: const Offset(1, 1), // 위치 조정
            ),
          ],
        ),
      ),
    );
  }

  /// 스크롤바 스타일 테마 정의
  ScrollbarThemeData _customScrollbarTheme() {
    return ScrollbarThemeData(
      trackVisibility: WidgetStateProperty.all(true), // 스크롤바 트랙 항상 표시
      trackColor:
          WidgetStateProperty.all(const Color(0xffbebebe)), // 스크롤바 트랙 색상 (회색)
      thumbColor: WidgetStateProperty.all(
          const Color.fromARGB(255, 255, 255, 255)), // 스크롤바 색상 (흰색)

      thickness: WidgetStateProperty.all(4), // 스크롤바 두께
      radius: const Radius.circular(10), // 스크롤바 모서리 둥글게 설정
    );
  }

  /// 팝업 제목
  Widget _buildTitle() {
    return Center(
      child: Text(
        widget.title, // 전달받은 제목 표시
        style: const TextStyle(fontSize: 17, fontWeight: FontWeight.bold),
      ),
    );
  }

  /// 이미지 (옵션)
  SizedBox _buildImage() {
    print('Image path: ${widget.imagePath}');
    // imagePath가 null이거나 빈 문자열인 경우 빈 위젯을 반환
    if (widget.imagePath == null || widget.imagePath!.isEmpty) {
      return const SizedBox.shrink(); // 이미지가 없으면 빈 위젯 return
    }

    return SizedBox(
      width: 300, // 이미지 너비 설정
      child: Image.asset(
        widget.imagePath!, // 전달받은 이미지 경로 사용
        fit: BoxFit.contain, // 이미지 비율 유지
      ),
    );
  }

  /// 설명 텍스트
  Padding _buildDescription() {
    return Padding(
      padding: const EdgeInsets.fromLTRB(10, 10, 30, 10), // 여백 추가
      child: Text(
        widget.description, // 전달받은 설명 텍스트 사용
        style: const TextStyle(fontSize: 14, color: Colors.black54),
        textAlign: TextAlign.center, // 가운데 정렬
      ),
    );
  }

  /// 닫기 버튼 (오른쪽 상단)
  Positioned _buildCloseButton(BuildContext context) {
    return Positioned(
      top: 8, // 상단 여백
      right: 8, // 오른쪽 여백
      child: IconButton(
        icon: const Icon(Icons.close), // 닫기 아이콘
        onPressed: () => Navigator.of(context).pop(), // 팝업 닫기 기능
      ),
    );
  }
}
