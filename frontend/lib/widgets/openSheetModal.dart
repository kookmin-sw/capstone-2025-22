import 'package:flutter/material.dart';
import 'package:flutter_inappwebview/flutter_inappwebview.dart';

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
}) async {
  // 2) 서버가 안 뜨면 올리기
  if (!localhostServer.isRunning()) {
    await localhostServer.start();
  }

  showDialog(
    context: context,
    builder: (_) => Dialog(
      insetPadding: EdgeInsets.symmetric(horizontal: 20, vertical: 0),
      backgroundColor: Colors.white,
      child: LayoutBuilder(
        builder: (context, constraints) {
          // 1) WebView 위젯 정의
          Widget webview = InAppWebView(
            initialUrlRequest: URLRequest(
              url: WebUri('http://localhost:8081/index.html'),
            ),
            onConsoleMessage: (controller, consoleMessage) {
              print("🖥️ [WebView Console] ${consoleMessage.message}");
            },
            onWebViewCreated: (ctrl) {
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
                  child: SizedBox(
                    width: constraints.maxWidth * 0.8,
                    height: constraints.maxHeight * 0.8,
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
                  right: 8,
                  child: IconButton(
                    icon: Icon(Icons.close_rounded,
                        size: 45, color: Colors.black54),
                    onPressed: () => Navigator.pop(context),
                  ),
                ),
              ],
            ),
          );
        },
      ),
    ),
  );
}
