import 'package:flutter_secure_storage/flutter_secure_storage.dart';

//Secure Storage를 앱 전역에서 사용
final FlutterSecureStorage storage = FlutterSecureStorage();

Future<bool> isStorageEmpty() async {
  // Secure Storage에 저장된 데이터가 있는지 확인
  final allKeys = await storage.readAll();
  return allKeys.isEmpty;
}
