import 'dart:convert';
import 'dart:typed_data';
import 'dart:ui';
import 'package:flutter/services.dart' show rootBundle;
import 'package:flutter_inappwebview/flutter_inappwebview.dart';

class OSMDService {
  InAppLocalhostServer localhostServer =
      InAppLocalhostServer(documentRoot: 'assets/web');
  HeadlessInAppWebView? headlessWebView; // 숨겨진 WebView

  void Function({
    required Uint8List base64Image,
    required Map<String, dynamic> json,
    required double bpm,
    required double canvasWidth,
    required double canvasHeight,
    required List<dynamic> lineBounds,
    required int totalMeasures,
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
          handlerName: 'getDataFromOSMD', // JS에서 요청하면 악보 렌더링 후 받은 데이터 보내줌
          callback: (args) {
            // args[0]이 Base64 String
            final String base64Image = args[0] as String;

            // args[1]이 info Map
            final Map<String, dynamic> info =
                Map<String, dynamic>.from(args[1] as Map);

            final double bpm = (info['bpm'] as num).toDouble();
            final double canvasWidth = (info['canvasWidth'] as num).toDouble();
            final double canvasHeight =
                (info['canvasHeight'] as num).toDouble();
            final int totalMeasures = (info['totalMeasures'] as num).toInt();
            final List<dynamic> lineBounds =
                info['lineBounds'] as List<dynamic>;

            onDataLoaded!(
              base64Image: base64Decode(base64Image), // String 타입이면
              json: info,
              bpm: bpm,
              canvasWidth: canvasWidth,
              canvasHeight: canvasHeight,
              lineBounds: lineBounds,
              totalMeasures: totalMeasures,
            );
          },
        );
      },
      onLoadStop: (controller, url) async {
        print("🛑 WebView Load 완료 (onLoadStop)");
        await Future.delayed(const Duration(milliseconds: 200));
        await controller.evaluateJavascript(source: 'startOSMDFromFlutter();');
      }, // HTML 로드 완료 후 JS 함수 실행해서 렌더링 시작
    );
    await headlessWebView?.run();
  }

  Future<void> dispose() async {
    await headlessWebView?.dispose();
    await localhostServer.close();
  }
}
