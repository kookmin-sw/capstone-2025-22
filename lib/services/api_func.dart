import 'package:capstone_2025/services/server_addr.dart'; //serverAddr
import 'package:http/http.dart' as http;
import 'dart:convert'; //jsonDecode

// HTTP handler 함수 - GET 요청 전용
Future<Map<String, dynamic>> getHTTP(
    String endpoint, Map<String, dynamic> queryParam) async {
  try {
    print("GET 요청 시작 --");

    final uri = Uri.http(
      serverAddr, // 서버 주소 : 현재 애뮬레이터 10.0.2.2:28080
      endpoint, // API 엔드포인트
      queryParam,
    );

    final response = await http.get(
      uri,
      headers: {
        "Accept": "application/json", // 기대하는 응답 형식
      },
    );

    print("_________________");
    print("응답 상태 코드: ${response.statusCode}");
    print("응답 본문: ${response.body}");
    print("_________________");

    if (response.statusCode == 200) {
      // 정상적인 경우
      final data = jsonDecode(response.body);
      data["errMessage"] = null; // 에러 메시지 초기화
      return data;
    } else {
      // 에러 코드별 세분화 처리
      switch (response.statusCode) {
        case 400:
          return {'errMessage': "잘못된 요청입니다. (정보 누락)"};
        case 401:
          return {'errMessage': "인증 실패: 잘못된 토큰 정보"};
        case 403:
          return {'errMessage': "권한이 없습니다."};
        case 404:
          return {'errMessage': "요청한 데이터를 찾을 수 없습니다."};
        case 500:
          return {'errMessage': "서버 내부 오류가 발생했습니다."};
        default:
          return {
            'errMessage': "알 수 없는 오류가 발생했습니다. (코드: ${response.statusCode})"
          };
      }
    }
  } catch (error) {
    // 네트워크 오류 또는 JSON 파싱 오류 처리
    return {'errMessage': "네트워크 오류가 발생했습니다."};
  }
}

// HTTP handler 함수 - POST 요청 전용
Future<Map<String, dynamic>> postHTTP(
    String endpoint, Map<String, dynamic> requestBody) async {
  try {
    print("POST 요청 시작 --");
    final uri = Uri.parse("http://$serverAddr$endpoint"); // URL 생성

    final response = await http.post(
      uri,
      headers: {
        "Content-Type": "application/json",
        "Accept": "application/json",
      },
      body: jsonEncode(requestBody),
    );

    print("_________________");
    print("응답 상태 코드: ${response.statusCode}");
    print("응답 본문: ${response.body}");
    print("_________________");

    if (response.statusCode == 200) {
      // 정상적인 경우
      final data = jsonDecode(response.body);
      data["errMessage"] = null; // 에러 메시지 초기화
      return data; // 사용자 정보 반환
    } else {
      // 에러 코드별 처리
      switch (response.statusCode) {
        case 400:
          return {'errMessage': "잘못된 요청입니다. (정보 누락)"};
        case 401:
          return {'errMessage': "인증 실패: 잘못된 토큰 정보"};
        case 403:
          return {'errMessage': "권한이 없습니다."};
        case 404:
          return {'errMessage': "사용자 정보를 찾을 수 없습니다."};
        case 409:
          return {'errMessage': "이미 존재하는 사용자입니다."};
        case 500:
          return {'errMessage': "서버 내부 오류가 발생했습니다."};
        default:
          return {
            'errMessage': "알 수 없는 오류가 발생했습니다. (코드: ${response.statusCode})"
          };
      }
    }
  } catch (error) {
    return {'errMessage': "네트워크 오류가 발생했습니다."};
  }
}

// HTTP handler 함수 - PUT 요청 전용
Future<Map<String, dynamic>> putHTTP(String endpoint,
    Map<String, dynamic>? queryParam, Map<String, dynamic> requestBody) async {
  try {
    print("PUT 요청 시작 --");

    final uri = queryParam != null
        ? Uri.http(serverAddr, endpoint, queryParam)
        : Uri.parse("http://$serverAddr$endpoint");

    final response = await http.put(
      uri,
      headers: {
        "Content-Type": "application/json",
        "Accept": "application/json",
      },
      body: jsonEncode(requestBody),
    );

    print("_________________");
    print("응답 상태 코드: ${response.statusCode}");
    print("응답 본문: ${response.body}");
    print("_________________");

    if (response.statusCode == 200) {
      final data = jsonDecode(response.body);
      data["errMessage"] = null; // 에러 메시지 초기화
      return data; // 정상 응답 반환
    } else {
      // 에러 코드별 처리
      switch (response.statusCode) {
        case 400:
          return {'errMessage': "잘못된 요청입니다. (정보 누락)"};
        case 401:
          return {'errMessage': "인증 실패: 잘못된 토큰 정보"};
        case 403:
          return {'errMessage': "권한이 없습니다."};
        case 500:
          return {'errMessage': "서버 내부 오류가 발생했습니다."};
        default:
          return {
            'errMessage': "알 수 없는 오류가 발생했습니다. (코드: ${response.statusCode})"
          };
      }
    }
  } catch (error) {
    return {'errMessage': "네트워크 오류가 발생했습니다."};
  }
}
