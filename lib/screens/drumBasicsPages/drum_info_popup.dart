import 'package:flutter/material.dart';

/// 팝업 창 위젯 파일
class DrumInfoPopup extends StatefulWidget {
  final String title;
  final String? imagePath;

  const DrumInfoPopup({
    super.key,
    required this.title, // 팝업 제목
    this.imagePath, // 이미지
  });

  @override
  State<DrumInfoPopup> createState() => _DrumInfoPopupState();
}

class _DrumInfoPopupState extends State<DrumInfoPopup> {
  final ScrollController _scrollController = ScrollController(); // 스크롤 컨트롤러

  @override
  void dispose() {
    _scrollController.dispose(); // 메모리 누수 방지
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final double screenHeight = MediaQuery.of(context).size.height; // 전체
    final double screenWidth = MediaQuery.of(context).size.width;

    return Dialog(
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12.0)),
      backgroundColor: Colors.white,
      child: SizedBox(
        width: screenWidth * 0.8, // 가로 80% 차지
        height: screenHeight, // 세로 전체 차지
        child: Stack(
          children: [
            Padding(
              padding: const EdgeInsets.fromLTRB(30, 30, 30, 60),
              child: ScrollbarTheme(
                data: ScrollbarThemeData(
                  trackVisibility:
                      WidgetStateProperty.all(true), // 스크롤바 배경 항상 표시
                  trackColor: WidgetStateProperty.all(
                      const Color(0xff949494)), // 스크롤바 배경 (회색)
                  thumbColor:
                      WidgetStateProperty.all(Colors.white), // 스크롤바 (흰색)
                  trackBorderColor: WidgetStateProperty.all(
                      const Color.fromARGB(255, 126, 126, 126)),
                  thickness: WidgetStateProperty.all(3), // 스크롤바 두께
                  radius: const Radius.circular(10), // 둥근 모서리
                ),
                child: Scrollbar(
                  controller: _scrollController,
                  trackVisibility: true, // 트랙(배경) 표시
                  thumbVisibility: true, // 스크롤바 표시
                  interactive: true, // 스크롤바 드래그 가능하게 설정
                  child: SingleChildScrollView(
                    controller: _scrollController,
                    physics: const AlwaysScrollableScrollPhysics(),
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        // 팝업 제목
                        Center(
                          child: Text(
                            widget.title,
                            style: const TextStyle(
                              fontSize: 17,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                        ),
                        const SizedBox(height: 20),

                        // 이미지
                        if (widget.imagePath != null)
                          SizedBox(
                            width: 300,
                            child: Image.asset(
                              widget.imagePath!,
                              fit: BoxFit.contain, // 이미지 비율 유지
                              // 주의: 이미지 크기가 고정되어 있으면 스크롤 불가
                            ),
                          ),

                        const SizedBox(height: 20),

                        // 스크롤 확인용 더미 텍스트
                        Padding(
                          padding: const EdgeInsets.all(10.0),
                          child: const Text(
                            "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Nisl tincidunt eget nullam non. Quis hendrerit dolor magna eget est lorem ipsum dolor sit. Volutpat odio facilisis mauris sit amet massa. Commodo odio aenean sed adipiscing diam donec adipiscing tristique. Mi eget mauris pharetra et. Non tellus orci ac auctor augue. Elit at imperdiet dui accumsan sit. Ornare arcu dui vivamus arcu felis. Egestas integer eget aliquet nibh praesent. In hac habitasse platea dictumst quisque sagittis purus. Pulvinar elementum integer enim neque volutpat ac.Senectus et netus et malesuada. Nunc pulvinar sapien et ligula ullamcorper malesuada proin. Neque convallis a cras semper auctor. Libero id faucibus nisl tincidunt eget. Leo a diam sollicitudin tempor id. A lacus vestibulum sed arcu non odio euismod lacinia. In tellus integer feugiat scelerisque. Feugiat in fermentum posuere urna nec tincidunt praesent. Porttitor rhoncus dolor purus non enim praesent elementum facilisis. Nisi scelerisque eu ultrices vitae auctor eu augue ut lectus. Ipsum faucibus vitae aliquet nec ullamcorper sit amet risus. Et malesuada fames ac turpis egestas sed. Sit amet nisl suscipit adipiscing bibendum est ultricies. Arcu ac tortor dignissim convallis aenean et tortor at. Pretium viverra suspendisse potenti nullam ac tortor vitae purus. Eros donec ac odio tempor orci dapibus ultrices. Elementum nibh tellus molestie nunc. Et magnis dis parturient montes nascetur. Est placerat in egestas erat imperdiet. Consequat interdum varius sit amet mattis vulputate enim.Sit amet nulla facilisi morbi tempus. Nulla facilisi cras fermentum odio eu. Etiam erat velit scelerisque in dictum non consectetur a erat. Enim nulla aliquet porttitor lacus luctus accumsan tortor posuere. Ut sem nulla pharetra diam. Fames ac turpis egestas maecenas. Bibendum neque egestas congue quisque egestas diam. Laoreet id donec ultrices tincidunt arcu non sodales neque. Eget felis eget nunc lobortis mattis aliquam faucibus purus. Faucibus interdum posuere lorem ipsum dolor sit.",
                            style:
                                TextStyle(fontSize: 14, color: Colors.black54),
                            textAlign: TextAlign.center,
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
            ),
            // 닫기 버튼 (오른쪽 상단)
            Positioned(
              top: 8,
              right: 8,
              child: IconButton(
                icon: const Icon(Icons.close),
                onPressed: () => Navigator.of(context).pop(),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
