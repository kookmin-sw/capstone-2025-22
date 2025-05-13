import 'dart:convert';
import 'package:stomp_dart_client/stomp_dart_client.dart';

class ScoringService {
  final StompClient client;
  String? _identifier; // 추가: identifier 상태로 저장

  ScoringService({required this.client});

  // identifier 저장하는 메서드 추가
  void setIdentifier(String identifier) {
    _identifier = identifier;
    print('✅ 연주 식별자 설정: $identifier');
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
      print('❌ identifier가 설정되지 않았습니다');
      return;
    }

    print('📤 마디 $measureNumber 오디오 데이터 전송');
    try {
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
        headers: {
          'content-type': 'application/json',
          'receipt': 'measure-$measureNumber',
        },
      );
    } catch (e) {
      print('❌ 오디오 데이터 전송 중 오류: $e');
    }
  }

// topicId를 받아서 '/topic/onset/$topicId' 구독, 해제함수 반환
  Function subscribeToScoringResults(
    String topicId,
    Function(Map<String, dynamic>) onResult,
  ) {
    final destination = '/topic/onset/$topicId';
    print('📥 채점 결과 구독 시작: $destination');

    final unsubscribeFn = client.subscribe(
      destination: destination,
      callback: (frame) {
        try {
          final result = jsonDecode(frame.body!);
          print('📥 채점 데이터 수신: $result');
          onResult(result);
        } catch (e) {
          print('❌ 채점 결과 처리 중 오류: $e');
        }
      },
    );
    return () {
      print('❌ 채점 결과 구독 해제: $destination');
      unsubscribeFn();
    };
  }
}
