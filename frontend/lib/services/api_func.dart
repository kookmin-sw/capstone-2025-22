import 'package:capstone_2025/services/server_addr.dart'; //serverAddr
import 'package:http/http.dart' as http;
import 'dart:convert'; //jsonDecode

// HTTP handler 함수 - GET 요청 전용
Future<Map<String, dynamic>> getHTTP(
    String endpoint, Map<String, dynamic> queryParam,
    {Map<String, dynamic> reqHeader = const {}}) async {
  try {
    print("GET 요청 시작 -- ${endpoint}");

    final uri = Uri.http(
      serverAddr, // 서버 주소
      endpoint, // API 엔드포인트
      queryParam,
    );

    final response = await http.get(
      uri,
      headers: {
        "Accept": "application/json", // 기대하는 응답 형식
        if (reqHeader != {}) ...reqHeader, // 추가 헤더 정보
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

Future<Map<String, dynamic>> postHTTP(String endpoint,
    Map<String, dynamic>? requestBody, // requestBody를 nullable로 변경
    {Map<String, dynamic> reqHeader = const {}}) async {
  try {
    print("POST 요청 시작 -- ${endpoint}");
    final uri = Uri.parse("http://$serverAddr$endpoint"); // URL 생성

    // requestBody가 null일 경우 빈 본문을 전달
    final body = requestBody != null ? jsonEncode(requestBody) : null;

    final response = await http.post(
      uri,
      headers: {
        "Content-Type": "application/json",
        "Accept": "application/json",
        if (reqHeader.isNotEmpty) ...reqHeader, // reqHeader가 있을 경우 추가
      },
      body: body, // requestBody가 null이면 본문을 보내지 않음
    );

    print("_________________");
    print("응답 상태 코드: ${response.statusCode}");
    print("응답 본문: ${response.body}");
    print("_________________");

    if (response.statusCode == 200) {
      // 정상적인 경우
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

Future<Map<String, dynamic>> putHTTP(
    String endpoint,
    Map<String, dynamic>? queryParam,
    Map<String, dynamic>? requestBody, // requestBody를 nullable로 변경
    {Map<String, dynamic> reqHeader = const {}}) async {
  try {
    print("PUT 요청 시작 -- ${endpoint}");

    // 쿼리 파라미터가 있을 경우 URI에 포함, 없으면 기본 endpoint로 URI 생성
    final uri = queryParam != null
        ? Uri.http(serverAddr, endpoint, queryParam)
        : Uri.parse("http://$serverAddr$endpoint");

    // requestBody가 null이 아니면 JSON으로 변환, null이면 본문을 비우기
    final body = requestBody != null ? jsonEncode(requestBody) : null;

    // PUT 요청 보내기
    final response = await http.put(
      uri,
      headers: {
        "Content-Type": "application/json", // 요청 본문 형식
        "Accept": "application/json", // 예상 응답 형식
        if (reqHeader.isNotEmpty) ...reqHeader, // reqHeader가 있을 경우 추가
      },
      body: body, // 본문이 null이면 빈 본문이 전달됨
    );

    print("_________________");
    print("응답 상태 코드: ${response.statusCode}");
    print("응답 본문: ${response.body}");
    print("_________________");

    // 상태 코드가 200이면 정상 응답 처리
    if (response.statusCode == 200) {
      final data = jsonDecode(response.body);
      data["errMessage"] = null; // 에러 메시지 초기화
      return data; // 정상 응답 데이터 반환
    } else {
      // 상태 코드별 오류 처리
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
    // 네트워크 오류나 다른 예외가 발생한 경우
    return {'errMessage': "네트워크 오류가 발생했습니다."};
  }
}
