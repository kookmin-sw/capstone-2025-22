import 'package:flutter/material.dart';
import 'package:flutter_inappwebview/flutter_inappwebview.dart';

void openMusicSheet({
  required BuildContext context,
  required String xmlDataString,
  required List<Map<String, dynamic>> practiceInfo,
}) {
  showDialog(
    context: context,
    builder: (_) => Dialog(
      insetPadding: EdgeInsets.symmetric(horizontal: 20, vertical: 0),
      backgroundColor: Colors.white,
      child: LayoutBuilder(
        builder: (context, constraints) {
          return SizedBox(
            width: constraints.maxWidth,
            height: constraints.maxHeight,
            child: Stack(
              children: [
                InAppWebView(
                  initialUrlRequest: URLRequest(
                      url: WebUri('http://localhost:8080/index.html')),
                  onWebViewCreated: (ctrl) {
                    // JS ⇄ Flutter 핸들러 등록
                    ctrl.addJavaScriptHandler(
                      handlerName: 'sendFileToOSMD',
                      callback: (_) => xmlDataString,
                    );
                    ctrl.addJavaScriptHandler(
                      handlerName: 'sendPracticeInfo',
                      callback: (_) => practiceInfo,
                    );
                  },
                  onLoadStop: (ctrl, url) async {
                    // JS 모듈(renderDetailedScore) 실행
                    await ctrl.evaluateJavascript(source: """
                      (async()=>{
                        const xml  = await window.flutter_inappwebview.callHandler('sendFileToOSMD');
                        const info = await window.flutter_inappwebview.callHandler('sendPracticeInfo');
                        await renderDetailedScore(xml, info, {
                          colorDefault: '#000000',
                          colorWrong1:   '#888888'
                        });
                      })();
                    """);
                  },
                ),
                IconButton(
                  onPressed: () => {Navigator.pop(context)},
                  icon: Icon(
                    Icons.close_rounded,
                    size: 45,
                    color: Colors.black54,
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
