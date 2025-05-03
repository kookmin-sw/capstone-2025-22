import 'dart:convert';
import 'dart:typed_data';
import 'dart:ui';
import 'package:flutter_inappwebview/flutter_inappwebview.dart';

class OSMDService {
  InAppLocalhostServer localhostServer =
      InAppLocalhostServer(documentRoot: 'assets/web');
  HeadlessInAppWebView? headlessWebView; // 숨겨진 WebView

  void Function({
    required String base64Image,
    required Map<String, dynamic> json,
    required double bpm,
    required double canvasWidth,
    required double canvasHeight,
  })? onDataLoaded; // JS에서 악보 이미지, 데이터 전달받으면 Flutter로 넘겨줄 콜백

  OSMDService({required this.onDataLoaded});

  Future<void> startOSMDService({
    required Uint8List xmlData,
    required double pageWidth,
    // List<ScoredEntry>? transcription, // 채점 기능 추가하면 필요함)
  }) async {
    if (localhostServer.isRunning()) {
      await localhostServer.close();
    }
    await localhostServer.start();

    // 기존 HeadlessWebView가 살아있으면 먼저 dispose 해줘야 함
    if (headlessWebView != null) {
      await headlessWebView!.dispose();
    }

    headlessWebView = HeadlessInAppWebView(
      initialUrlRequest: URLRequest(url: WebUri('http://localhost:8080')),
      initialSize: Size(1080, 0), // 가로 고정
      onWebViewCreated: (controller) async {
        await InAppWebViewController.clearAllCache();
        await controller.clearCache();
        controller.addJavaScriptHandler(
            handlerName: 'sendFileToOSMD',
            callback: (_) async {
              // JS쪽에서 MusicXML 데이터 요청하면 넘겨줌
              return utf8.decode(xmlData); // utf8 디코딩된 XML
            });

        controller.addJavaScriptHandler(
          handlerName: 'getDataFromOSMD', // JS에서 요청하면 악보 렌더링 후 받은 데이터 보내줌줌
          callback: (args) {
            // args[0]이 Base64 String
            final String base64Image = args[0] as String;
            final Uint8List imageBytes = base64Decode(base64Image);

            // args[1]이 info Map
            final Map<String, dynamic> info =
                Map<String, dynamic>.from(args[1] as Map);

            final double bpm = (info['bpm'] as num).toDouble();
            final double canvasWidth = (info['canvasWidth'] as num).toDouble();
            final double canvasHeight =
                (info['canvasHeight'] as num).toDouble();
            final List<dynamic> cursorJson =
                info['cursorList'] as List<dynamic>;

            print(
                '📌 [DART/OSMDService] cursorJson length: ${cursorJson.length}');
            print(
                '📌 [DART/OSMDService] sample cursorJson[0]: ${cursorJson[0]}');

            if (onDataLoaded != null) {
              onDataLoaded!(
                base64Image: base64Image, // 혹은 imageBytes로 바꿔 넘겨도 됨
                json: info,
                bpm: bpm,
                canvasWidth: canvasWidth,
                canvasHeight: canvasHeight,
              );
            }
          },
        );
      },
      onLoadStop: (controller, url) async {
        print("🛑 WebView Load 완료 (onLoadStop)");
        await Future.delayed(const Duration(milliseconds: 200));
        await controller.evaluateJavascript(source: 'startOSMDFromFlutter();');
      }, // HTML 로드 완료 후 JS 함수 실행해서 렌더링 시작작
    );
    await headlessWebView?.run();
  }

  Future<void> dispose() async {
    await headlessWebView?.dispose();
    await localhostServer.close();
  }
}

/***
 assets/web/index.html 로드할 수 있도록 로컬 서버 띄우고
 headlessWebview로 화면에 보이지 않고 렌더링만 수행하는 용도
  ***/
