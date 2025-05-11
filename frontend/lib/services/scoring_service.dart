import 'dart:convert';
import 'package:stomp_dart_client/stomp_dart_client.dart';

class ScoringService {
  final StompClient client;
  String? _identifier; // 추가: identifier 상태로 저장

  ScoringService({required this.client});

  // identifier 저장하는 메서드 추가
  void setIdentifier(String identifier) {
    _identifier = identifier;
  }

  String get identifier {
    if (_identifier == null) {
      throw Exception("Identifier가 설정되지 않았습니다.");
    }
    return _identifier!;
  }

  void sendMeasureAudio({
    required String email,
    required String base64Wav,
    required int bpm,
    required int userSheetId,
    required String measureNumber,
    required bool endOfMeasure,
  }) {
    if (_identifier == null) {
      throw Exception("Identifier가 설정되지 않았습니다.");
    }

    client.send(
      destination: '/app/audio/forwarding',
      body: jsonEncode({
        "bpm": bpm,
        "userSheetId": userSheetId,
        "identifier": _identifier,
        "email": email,
        "message": base64Wav,
        "measureNumber": measureNumber,
        "endOfMeasure": endOfMeasure
      }),
      headers: {'content-type': 'application/json'},
    );
  }

  void subscribeToScoringResults(
      String email, Function(Map<String, dynamic>) onResult) {
    client.subscribe(
      destination: '/topic/onset/$email',
      callback: (frame) {
        final result = jsonDecode(frame.body!);
        onResult(result);
      },
    );
  }
}
