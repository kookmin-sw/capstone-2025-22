import 'package:flutter_secure_storage/flutter_secure_storage.dart';

//Secure Storage를 앱 전역에서 사용
final FlutterSecureStorage storage = FlutterSecureStorage();

Future<bool> isStorageEmpty() async {
  // Secure Storage에 저장된 데이터가 있는지 확인
  final allKeys = await storage.readAll();
  return allKeys.isEmpty;
}

// 코치마크 최초 실행 여부 조회 (없으면 false 반환)
Future<bool> hasShownCoachMark() async {
  String? value = await storage.read(key: 'musicsheet_coachmark_shown');
  return value == 'true';
}

// 코치마크 실행 완료 여부 저장
Future<void> setShownCoachMark() async {
  await storage.write(key: 'musicsheet_coachmark_shown', value: 'true');
}
