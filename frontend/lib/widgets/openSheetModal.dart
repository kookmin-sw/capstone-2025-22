import 'package:flutter/material.dart';
import 'package:flutter_inappwebview/flutter_inappwebview.dart';
import 'package:flutter_screenutil/flutter_screenutil.dart';

// 1) 앱 내부에서 assets/web 폴더를 서비스할 서버
final localhostServer = InAppLocalhostServer(
  documentRoot: 'assets/web',
  port: 8081,
);

Future<void> openMusicSheet({
  required BuildContext context,
  required String xmlDataString,
  required List<Map<String, dynamic>> practiceInfo,
  bool isPatternMode = false,
  bool isResultPage = true,
}) async {
  OverlayEntry? _infoOverlayEntry;

  void showInfoOverlay() {
    final overlay = Overlay.of(context);
    if (overlay != null) {
      final overlayEntry = OverlayEntry(
        builder: (context) => Positioned(
          top: MediaQuery.of(context).size.height * 0.12,
          left: MediaQuery.of(context).size.width * 0.075,
          child: Material(
            color: Colors.transparent,
            child: Container(
              alignment: Alignment.center,
              padding: EdgeInsets.symmetric(vertical: 20, horizontal: 30),
              decoration: BoxDecoration(
                color: Color(0xfff5f5f5),
                borderRadius: BorderRadius.circular(12),
                boxShadow: [
                  BoxShadow(
                    color: Colors.black45,
                    blurRadius: 10,
                    offset: Offset(0, 4),
                  ),
                ],
              ),
              child: isResultPage
                  ? Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          "아래는 1차 채점 결과입니다.",
                          style: TextStyle(
                              color: Color(0xff595959),
                              fontSize: 8.5.sp,
                              fontWeight: FontWeight.w600),
                        ),
                        Row(
                          children: [
                            Icon(Icons.circle, color: Colors.black, size: 20),
                            Text(
                              " : 박자 정답",
                              style: TextStyle(
                                  color: Color(0xff595959),
                                  fontSize: 7.5.sp,
                                  fontWeight: FontWeight.w600),
                            )
                          ],
                        ),
                        Row(
                          children: [
                            Icon(Icons.circle,
                                color: Color(0xffb2b2b2), size: 20),
                            Text(
                              " : 박자 오답",
                              style: TextStyle(
                                  color: Color(0xff595959),
                                  fontSize: 7.5.sp,
                                  fontWeight: FontWeight.w600),
                            )
                          ],
                        ),
                      ],
                    )
                  : Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          "아래는 최종 채점 결과입니다.",
                          style: TextStyle(
                              color: Color(0xff595959),
                              fontSize: 8.5.sp,
                              fontWeight: FontWeight.w600),
                        ),
                        Row(
                          children: [
                            Icon(Icons.circle, color: Colors.black, size: 20),
                            Text(
                              " : 1차·2차 모두 정답",
                              style: TextStyle(
                                  color: Color(0xff595959),
                                  fontSize: 7.5.sp,
                                  fontWeight: FontWeight.w600),
                            )
                          ],
                        ),
                        Row(
                          children: [
                            Icon(Icons.circle,
                                color: Color(0xfff5260f), size: 20),
                            Text(
                              " : 1차·2차 모두 오답",
                              style: TextStyle(
                                  color: Color(0xff595959),
                                  fontSize: 7.5.sp,
                                  fontWeight: FontWeight.w600),
                            )
                          ],
                        ),
                        Row(
                          children: [
                            Icon(Icons.circle,
                                color: Color(0xfff5a00f), size: 20),
                            Text(
                              " : 1차 오답 / 2차 정답",
                              style: TextStyle(
                                  color: Color(0xff595959),
                                  fontSize: 7.5.sp,
                                  fontWeight: FontWeight.w600),
                            )
                          ],
                        ),
                        Row(
                          children: [
                            Icon(Icons.circle,
                                color: Color(0xfff5e90f), size: 20),
                            Text(
                              " : 1차 정답 / 2차 오답",
                              style: TextStyle(
                                  color: Color(0xff595959),
                                  fontSize: 7.5.sp,
                                  fontWeight: FontWeight.w600),
                            )
                          ],
                        )
                      ],
                    ),
            ),
          ),
        ),
      );
      overlay.insert(overlayEntry);
      _infoOverlayEntry = overlayEntry;
    }
  }

  void removeInfoOverlay() {
    _infoOverlayEntry?.remove();
    _infoOverlayEntry = null;
  }

  // 2) 서버가 안 뜨면 올리기
  if (!localhostServer.isRunning()) {
    await localhostServer.start();
  }

  showDialog(
    context: context,
    builder: (_) => StatefulBuilder(
      builder: (context, setState) {
        return Dialog(
          insetPadding: EdgeInsets.symmetric(horizontal: 10, vertical: 20),
          backgroundColor: Colors.white,
          child: LayoutBuilder(
            builder: (context, constraints) {
              // 1) WebView 위젯 정의
              Widget webview = InAppWebView(
                initialUrlRequest: URLRequest(
                  url: WebUri('http://localhost:8081/index.html'),
                ),
                onWebViewCreated: (ctrl) async {
                  // XML 전송 핸들러
                  ctrl.addJavaScriptHandler(
                    handlerName: 'sendFileToOSMD',
                    callback: (_) => xmlDataString,
                  );
                  // practiceInfo 전송 핸들러
                  ctrl.addJavaScriptHandler(
                    handlerName: 'sendPracticeInfo',
                    callback: (_) => practiceInfo,
                  );
                },
                onLoadStop: (ctrl, url) async {
                  await ctrl.evaluateJavascript(source: """
                          (async()=>{
                            const xml  = await window.flutter_inappwebview.callHandler('sendFileToOSMD');
                            const info = await window.flutter_inappwebview.callHandler('sendPracticeInfo');
                            await renderDetailedScore(xml, info, {
                              colorDefault:    '#000000',
                              colorWrong1:     '#b2b2b2',
                              colorBothWrong:  '#f5260f',
                              colorWrong1Only: '#f5a00f',
                              colorWrong2Only: '#f5e90f'
                            });
                          })();
                        """);
                },
              );
              // 모드에 따라 가운데 정렬 or 풀스크린
              Widget content = isPatternMode
                  ? Center(
                      child: FractionallySizedBox(
                        widthFactor: 1.0,
                        heightFactor: 0.5,
                        child: webview,
                      ),
                    )
                  : SizedBox.expand(child: webview);
              // 3) content + close 버튼을 Stack으로 합치기
              return SizedBox(
                width: constraints.maxWidth,
                height: constraints.maxHeight,
                child: Stack(
                  children: [
                    content,
                    Positioned(
                      top: 8,
                      left: 8,
                      right: 8,
                      child: Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          GestureDetector(
                            onTapDown: (_) {
                              showInfoOverlay();
                            },
                            onTapUp: (_) {
                              removeInfoOverlay();
                            },
                            onTapCancel: () {
                              removeInfoOverlay();
                            },
                            child: Icon(Icons.info,
                                size: 45, color: Colors.black54),
                          ),
                          IconButton(
                            icon: Icon(Icons.close_rounded,
                                size: 45, color: Colors.black54),
                            onPressed: () => Navigator.pop(context),
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
              );
            },
          ),
        );
      },
    ),
  );
}
