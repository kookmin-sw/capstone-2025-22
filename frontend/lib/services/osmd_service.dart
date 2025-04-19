// lib/services/osmd_service.dart

import 'package:flutter/services.dart';
import 'package:flutter_inappwebview/flutter_inappwebview.dart';

class OSMDService {
  /// JS 핸들러 등록 (각 WebViewController에 따로 적용)
  Future<void> initHandlers({
    required InAppWebViewController controller,
    Function(double x)? onCursorStep,
  }) async {
    controller.addJavaScriptHandler(
      handlerName: 'sendFileToOSMD',
      callback: (args) async {
        String xml = await rootBundle.loadString('assets/music/demo.xml');
        return xml;
      },
    );

    controller.addJavaScriptHandler(
      handlerName: 'onCursorStep',
      callback: (args) {
        final cursorData = args[0];
        print("🟠 onCursorStep received: $cursorData");
        if (onCursorStep != null &&
            cursorData != null &&
            cursorData['x'] != null) {
          onCursorStep(cursorData['x'].toDouble());
        }
        return null;
      },
    );
  }

  /// 특정 줄 로드
  Future<void> loadLine(
      InAppWebViewController controller, int lineNumber) async {
    await controller.evaluateJavascript(source: 'loadLine($lineNumber);');
  }

  /// 다음 커서 위치로 이동
  Future<void> moveNextCursorStep(InAppWebViewController controller) async {
    await controller.evaluateJavascript(source: 'moveNextCursorStep();');
  }

  /// 전체 줄 수 가져오기
  Future<int> getTotalLineCount(InAppWebViewController controller) async {
    final result =
        await controller.evaluateJavascript(source: 'getTotalLineCount();');
    return int.tryParse(result.toString()) ?? 1;
  }

  /// 현재 줄의 음표 개수 가져오기
  Future<int> getNoteCountForLine(
      InAppWebViewController controller, int line) async {
    final result = await controller.evaluateJavascript(
        source: 'getNoteCountForLine($line);');
    return int.tryParse(result.toString()) ?? 0;
  }
}
