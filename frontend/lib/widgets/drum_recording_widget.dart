// ignore_for_file: avoid_print
import 'dart:io';
import 'dart:async';
import 'dart:convert';
import 'package:xml/xml.dart';
import 'package:logger/logger.dart';
import 'package:flutter/services.dart';
import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';
import 'package:flutter_sound/flutter_sound.dart' as fs;
import 'package:stomp_dart_client/stomp_dart_client.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:flutter_sound/public/flutter_sound_recorder.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import 'package:capstone_2025/screens/drumSheetPages/playback_controller.dart';

/// 드럼 녹음 기능을 제공하는 위젯
/// 카운트다운, WebSocket 연결, XML 파싱, 녹음 등의 기능을 포함
class DrumRecordingWidget extends StatefulWidget {
  /// 화면 상단에 표시될 제목
  final String title;

  /// MusicXML 파일 경로 혹은 파일 내용
  final String? xmlFilePath; // 패턴 및 필인 페이지에서 사용
  final String? xmlDataString; // 악보 연습 페이지에서 사용

  /// 연습용 오디오 파일 경로
  final String audioFilePath;

  /// 녹음 완료 시 호출될 콜백 함수
  final Function(List<dynamic>)? onRecordingComplete;

  /// 현재 마디 정보 업데이트를 위한 콜백
  final Function(int currentMeasure, int totalMeasures)? onMeasureUpdate;

  /// 온셋 데이터 수신 시 호출될 콜백
  final Function(List<dynamic> onsets)? onOnsetsReceived;

  /// MusicXML 파싱 결과를 부모 위젯에 전달하기 위한 콜백
  final Function(Map<String, dynamic>)? onMusicXMLParsed;

  /// 배속 정보
  final PlaybackController playbackController;

  const DrumRecordingWidget({
    super.key,
    required this.title,
    this.xmlFilePath,
    this.xmlDataString,
    required this.audioFilePath,
    this.onRecordingComplete,
    this.onMeasureUpdate,
    this.onOnsetsReceived,
    this.onMusicXMLParsed,
    required this.playbackController,
  });

  @override
  State<DrumRecordingWidget> createState() => DrumRecordingWidgetState();
}

class DrumRecordingWidgetState extends State<DrumRecordingWidget>
    with SingleTickerProviderStateMixin {
  // WebSocket 관련
  StompClient? _stompClient;
  bool _webSocketConnected = false;
  int _reconnectAttemps = 0;
  final int _maxReconnectAttempts = 5;
  String _userEmail = '';
  final _storage = const FlutterSecureStorage();
  Function? _stompUnsubscribe;
  bool _isDisposed = false;

  // 녹음 관련
  bool isRecording = false;
  String? _recordingPath;
  fs.FlutterSoundRecorder? _recorder;
  String recordingStatusMessage = '';
  Timer? _recordingTimer;
  StreamSubscription<fs.RecordingDisposition>? _recorderSubscription;

  // XML 파싱 및 타이밍 관련
  int _beatsPerMeasure = 4;
  int _beatType = 4;
  int _totalMeasures = 1;
  final double _baseBpm = 60.0;
  double _totalDuration = 0.0;
  int _currentMeasure = 0;
  double _secondsPerMeasure = 0.0; // 한 마디당 시간(초), XML 파싱 후 계산됨

  // 카운트다운 관련
  int countdown = 3;
  bool isCountingDown = false;
  Timer? _countdownTimer;
  late AnimationController _overlayController;
  late Animation<double> _overlayAnimation;

  // 결과
  List<dynamic> _detectedOnsets = [];

  @override
  void initState() {
    super.initState();

    // 오버레이 애니메이션 초기화
    _overlayController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 500),
    );
    _overlayAnimation =
        Tween<double>(begin: 0.0, end: 1.0).animate(_overlayController);

    // 데이터 초기화
    _initializeData();
    _parseMusicXML();

    // PlaybackController의 이벤트 구독
    widget.playbackController.onMeasureChange = _handleMeasureChange;
    widget.playbackController.onCountdownComplete = _handleCountdownComplete;
    widget.playbackController.onPlaybackComplete = _handlePlaybackComplete;
  }

  Future<void> _initializeData() async {
    if (_isDisposed) return;

    // 저장된 이메일 불러오기
    _userEmail = await _storage.read(key: 'user_email') ?? 'test@example.com';

    // 녹음기 초기화
    _recorder = fs.FlutterSoundRecorder();
    await _initRecorder();

    // WebSocket 연결 설정
    await _setupWebSocket();
  }

  Future<void> _initRecorder() async {
    if (_isDisposed) return;

    // 마이크 권한 요청
    var status = await Permission.microphone.request();
    if (status != PermissionStatus.granted) {
      throw RecordingPermissionException('마이크 권한이 부여되지 않았습니다.');
    }

    // 녹음기 관련 로그 끄기
    _recorder = fs.FlutterSoundRecorder(logLevel: Level.off);

    await _recorder?.openRecorder();

    // 녹음 파일 저장 경로 설정
    final appDocDir = await getApplicationDocumentsDirectory();
    _recordingPath = '${appDocDir.path}/current_measure.wav';
  }

  Future<void> _setupWebSocket() async {
    if (_isDisposed) return;

    final token = await _storage.read(key: 'access_token');
    print('🔑 WebSocket 연결 시도 - 토큰: $token');

    // WebSocket 설정
    _stompClient = StompClient(
      config: StompConfig.sockJS(
        url: 'http://34.68.164.98:28080/ws/audio',
        onConnect: (StompFrame frame) {
          if (_isDisposed) return;

          print('✅ WebSocket 연결 완료!');
          setState(() {
            _webSocketConnected = true;
          });
          _reconnectAttemps = 0;
          _subscribeToTopic();
        },
        beforeConnect: () async => print('🌐 WebSocket 연결 시도 중...'),
        onWebSocketError: (dynamic error) {
          print('❌ WebSocket 오류 발생: $error');
          if (!_isDisposed) {
            _retryWebSocketConnect();
          }
        },
        onDisconnect: (frame) {
          print('🔌 WebSocket 연결 끊어짐');
          if (!_isDisposed) {
            setState(() {
              _webSocketConnected = false;
            });
          }
        },
        stompConnectHeaders: {
          'Authorization': token ?? '',
        },
      ),
    );

    if (!_isDisposed) {
      _stompClient?.activate();
    }
  }

  void _subscribeToTopic() {
    if (_isDisposed || _stompClient == null) return;

    _stompUnsubscribe = _stompClient!.subscribe(
      destination: '/topic/onset/$_userEmail',
      callback: (frame) {
        if (_isDisposed) return;

        if (frame.body != null) {
          final response = json.decode(frame.body!);
          print('📦 WebSocket 데이터 수신 완료: $response');

          if (response.containsKey('onsets')) {
            if (!_isDisposed) {
              setState(() {
                _detectedOnsets = response['onsets'];
              });
            }
            print('🎯 감지된 온셋 수: ${response['onsets']}');

            // 부모 위젯에 콜백으로 알림
            if (widget.onOnsetsReceived != null && !_isDisposed) {
              widget.onOnsetsReceived!(_detectedOnsets);
            }
          }
        } else {
          print('⚠️ 빈 WebSocket 프레임 수신');
        }
      },
    );
  }

  void _retryWebSocketConnect() {
    if (_isDisposed) return;

    if (_reconnectAttemps < _maxReconnectAttempts) {
      _reconnectAttemps++;
      Future.delayed(const Duration(seconds: 3), () {
        if (_isDisposed) return;

        print(
            '🔁 WebSocket 재연결 시도 ($_reconnectAttemps/$_maxReconnectAttempts)...');
        _stompClient?.activate();
      });
    } else {
      print('❌ WebSocket 재연결 실패 - 최대 시도 초과');
    }
  }

  /// 마디 타이밍 정보 설정
  void setMeasureInfo(Map<String, dynamic> info) {
    if (_isDisposed) return;

    setState(() {
      _beatsPerMeasure = info['beatsPerMeasure'] as int;
      _beatType = info['beatType'] as int;
      _totalMeasures = info['totalMeasures'] as int;
      _totalDuration = info['totalDuration'] as double;
      _secondsPerMeasure = info['secondsPerMeasure'] as double;
    });

    print('✅ DrumRecordingWidget: 마디 정보 업데이트 완료');
  }

  Future<void> _parseMusicXML() async {
    print('🔍 _parseMusicXML 호출');
    if (_isDisposed) return;

    try {
      String xmlDataString;

      // xmlFilePath가 주어지면 파일을 읽어와서 xmlDataString으로 사용
      if (widget.xmlFilePath != null) {
        print('🔍 xmlFilePath 존재');
        xmlDataString = await rootBundle.loadString(widget.xmlFilePath!);
        print('🔍 xmlDataString: $xmlDataString');
      }
      // xmlDataString이 주어지면 그대로 사용
      else if (widget.xmlDataString != null) {
        print('🔍 xmlDataString 존재');
        xmlDataString = widget.xmlDataString!;
        print('🔍 XML Data: $xmlDataString');
      } else {
        print("❌ xmlDataString 또는 xmlFilePath가 제공되지 않았습니다.");
        return;
      }

      // XML 선언 추가 (만약 없다면)
      if (!xmlDataString.startsWith('<?xml')) {
        xmlDataString =
            '<?xml version="1.0" encoding="UTF-8"?>\n$xmlDataString';
      }

      // XML 파싱
      final document = XmlDocument.parse(xmlDataString);

      // 박자 정보 파싱
      final timeElement = document.findAllElements('time').first;
      _beatsPerMeasure =
          int.parse(timeElement.findElements('beats').first.text);
      _beatType = int.parse(timeElement.findElements('beat-type').first.text);

      // 총 마디 수 계산
      _totalMeasures = document.findAllElements('measure').length;

      // BPM 추출
      double? parsedBpm;
      final soundElem = document.findAllElements('sound').firstOrNull;
      if (soundElem != null && soundElem.getAttribute('tempo') != null) {
        parsedBpm = double.tryParse(soundElem.getAttribute('tempo')!);
      }
      if (parsedBpm == null) {
        final perMinuteElem =
            document.findAllElements('per-minute').firstOrNull;
        if (perMinuteElem != null) {
          parsedBpm = double.tryParse(perMinuteElem.text);
        }
      }
      if (parsedBpm == null) {
        final bpmElem = document.findAllElements('bpm').firstOrNull;
        if (bpmElem != null) {
          parsedBpm = double.tryParse(bpmElem.text);
        }
      }
      final bpm = parsedBpm ?? 60.0;

      // 한 마디당 시간 계산 (초)
      _secondsPerMeasure = (_beatsPerMeasure * 60.0) / bpm;

      // 총 재생 시간 계산 (초)
      _totalDuration = _totalMeasures * _secondsPerMeasure;

      print('🎼≪MusicXML 파싱 결과≫🎼');
      print('박자: $_beatsPerMeasure/$_beatType');
      print('총 마디 수: $_totalMeasures');
      print('BPM: $bpm');
      print('한 마디 시간: ${_secondsPerMeasure.toStringAsFixed(2)}초');
      print('총 재생 시간: ${_totalDuration.toStringAsFixed(2)}초');

      // 부모 위젯에 파싱 결과 전달
      if (widget.onMusicXMLParsed != null && !_isDisposed) {
        widget.onMusicXMLParsed!({
          'beatsPerMeasure': _beatsPerMeasure,
          'beatType': _beatType,
          'totalMeasures': _totalMeasures, // 제대로 설정된 totalMeasures
          'bpm': bpm,
          'totalDuration': _totalDuration,
          'secondsPerMeasure': _secondsPerMeasure,
        });
      }
    } catch (e) {
      print('❌ MusicXML 파싱 오류: $e');
    }
  }

  /// 카운트다운 애니메이션 시작
  void startCountdown({Function? onCountdownComplete}) {
    if (!mounted || _isDisposed) return;

    setState(() {
      isCountingDown = true;
      countdown = 3; // 3초 카운트다운 시작
    });

    _overlayController.forward(); // 카운트다운 페이드 인

    _countdownTimer = Timer.periodic(const Duration(seconds: 1), (timer) {
      if (!mounted || _isDisposed) {
        timer.cancel();
        return;
      }

      if (countdown == 0) {
        timer.cancel();
        _overlayController.reverse().then((_) async {
          if (!mounted || _isDisposed) return;
          setState(() => isCountingDown = false);

          // 카운트다운 완료 시 콜백 호출
          if (onCountdownComplete != null) {
            onCountdownComplete();
          }
        });
      } else {
        setState(() => countdown--);
      }
    });
  }

  // 카운트다운 완료 처리
  void _handleCountdownComplete() {
    if (!mounted || _isDisposed) return;
    startRecording();
  }

  // 연주 완료 처리
  void _handlePlaybackComplete(int lastMeasure) {
    if (!isRecording || _isDisposed) return;
    print('🎼 연주 완료 감지: 마지막 마디 $lastMeasure');

    // 1. 마지막 마디 녹음 중지 (데이터 저장)
    if (_recorder!.isRecording) {
      _recorder!.stopRecorder().then((_) {
        print('🎙️ 마지막 마디 녹음 중지 완료');
        // 2. 저장된 데이터 전송
        return _sendRecordingData();
      }).then((_) {
        print('📤 마지막 마디 녹음 데이터 전송 완료: ${DateTime.now()}');
        // 3. 녹음 종료
        return stopRecording();
      }).then((_) {
        print('🎙️ 녹음 종료 완료');
      }).catchError((error) {
        print('❌ 마지막 마디 처리 중 오류 발생: $error');
      });
    } else {
      print('⚠️ 마지막 마디 녹음이 이미 중지된 상태입니다');
      stopRecording();
    }
  }

  /// 오디오 녹음 시작
  void startRecording() async {
    if (isRecording || !mounted || _isDisposed || _recorder == null) return;

    // WebSocket 연결 확인
    if (!_webSocketConnected) {
      print('❌ 녹음을 시작할 수 없습니다: WebSocket이 연결되지 않았습니다.');
      setState(() => recordingStatusMessage = 'WebSocket 연결이 필요합니다!');
      return;
    }

    try {
      // 전체 녹음 프로세스 시작
      setState(() {
        isRecording = true;
        _currentMeasure = 0;
        recordingStatusMessage = '녹음이 시작되었습니다.';
      });

      // 첫 마디 녹음 시작
      await _startMeasureRecording();
    } catch (e) {
      if (!_isDisposed) {
        setState(() => recordingStatusMessage = '녹음 시작 실패: $e');
      }
      print('❌ 녹음 중 오류 발생: $e');
    }
  }

  /// 오디오 녹음 중지
  Future<void> stopRecording() async {
    if (!isRecording || !mounted || _isDisposed || _recorder == null) return;

    try {
      if (_recorder!.isRecording) {
        await _recorder!.stopRecorder(); // 현재 진행 중인 녹음 중지
      }

      if (!_isDisposed) {
        setState(() {
          isRecording = false;
          recordingStatusMessage = '녹음이 중지되었습니다.';
        });
      }

      // 부모 위젯에 결과 전달
      if (widget.onRecordingComplete != null && !_isDisposed) {
        widget.onRecordingComplete!(_detectedOnsets);
      }
    } catch (e) {
      print('❌ 녹음 종료 중 오류 발생: $e');
    }
  }

  /// 카운트다운 오버레이 위젯 반환
  Widget buildCountdownOverlay() {
    if (!isCountingDown) return const SizedBox.shrink();

    return FadeTransition(
      opacity: _overlayAnimation,
      child: Container(
        color: Colors.black.withOpacity(0.6),
        alignment: Alignment.center,
        child: Row(
          mainAxisAlignment: MainAxisAlignment.spaceEvenly,
          children: List.generate(3, (i) {
            int number = 3 - i;
            return Padding(
              padding: const EdgeInsets.symmetric(horizontal: 32),
              child: Stack(
                alignment: Alignment.center,
                children: [
                  Text(
                    '$number',
                    style: TextStyle(
                      fontSize: 72,
                      fontWeight: FontWeight.bold,
                      foreground: Paint()
                        ..style = PaintingStyle.stroke
                        ..strokeWidth = 10
                        ..color = countdown == number
                            ? const Color(0xffB95D4C)
                            : const Color(0xff949494),
                    ),
                  ),
                  Text(
                    '$number',
                    style: TextStyle(
                      fontSize: 72,
                      fontWeight: FontWeight.bold,
                      color: countdown == number
                          ? const Color(0xffFD9B8A)
                          : const Color(0xfff6f6f6),
                    ),
                  ),
                ],
              ),
            );
          }),
        ),
      ),
    );
  }

  /// 마디 타이밍 정보 반환
  Map<String, dynamic> getMeasureInfo() {
    return {
      'beatsPerMeasure': _beatsPerMeasure,
      'beatType': _beatType,
      'totalMeasures': _totalMeasures,
      'totalDuration': _totalDuration,
      'secondsPerMeasure': _secondsPerMeasure,
    };
  }

  /// 모든 리소스를 안전하게 정리하는 메서드
  void cleanupResources() async {
    print('🧹 리소스 정리 시작...');

    // 1. 모든 타이머 취소
    _countdownTimer?.cancel();
    _recordingTimer?.cancel();

    // 2. 구독 취소
    _recorderSubscription?.cancel();
    if (_stompUnsubscribe != null) {
      try {
        _stompUnsubscribe!();
        print('✅ WebSocket 구독 취소 완료');
      } catch (e) {
        print('⚠️ WebSocket 구독 취소 중 오류 발생: $e');
      }
      _stompUnsubscribe = null;
    }

    // 3. 녹음 중지
    if (isRecording && _recorder != null) {
      try {
        if (_recorder!.isRecording) {
          await _recorder!.stopRecorder();
        }
        print('✅ 녹음 중지 완료');
      } catch (e) {
        print('⚠️ 녹음 중지 중 오류 발생: $e');
      }
    }

    // 4. 녹음기 정리
    if (_recorder != null) {
      try {
        await _recorder!.closeRecorder();
        print('✅ 녹음기 종료 완료');
      } catch (e) {
        print('⚠️ 녹음기 종료 중 오류 발생: $e');
      }
      _recorder = null;
    }

    // 5. WebSocket 연결 종료
    if (_stompClient != null) {
      try {
        // 연결 상태 확인 없이 무조건 비활성화 시도
        _stompClient!.deactivate();
        print('✅ WebSocket 종료 완료');
      } catch (e) {
        print('⚠️ WebSocket 종료 중 오류 발생: $e');
      }
      _stompClient = null;
    }

    print('✅ 모든 리소스 정리 완료');
  }

  // 마디 변경 처리
  void _handleMeasureChange(int measureNumber) {
    if (!isRecording || _isDisposed) return;

    // 첫 번째 마디 변경 감지인 경우 (녹음 시작)
    if (_currentMeasure == 0 && measureNumber == 0) {
      _startMeasureRecording();
      return;
    }

    // 현재 마디 녹음 중지 및 데이터 전송
    _processCurrentMeasure();
  }

  // 마디 단위 처리
  Future<void> _processCurrentMeasure() async {
    if (!isRecording || _isDisposed || _recorder == null) return;

    try {
      // 현재 녹음 중지
      if (_recorder!.isRecording) {
        await _recorder!.stopRecorder();
      }

      // 녹음 데이터 전송
      await _sendRecordingData();
      print('📤 마디 ${_currentMeasure + 1} 녹음 데이터 전송 완료: ${DateTime.now()}');

      // 다음 마디 녹음 시작 (첫 마디가 아닌 경우에만)
      if (_currentMeasure < _totalMeasures - 1) {
        _currentMeasure++;
        if (_currentMeasure > 0) {
          // 첫 마디가 아닌 경우에만 녹음 시작
          await _startMeasureRecording();
        }
      }
    } catch (e) {
      print('❌ 마디 처리 중 오류 발생: $e');
      if (!_isDisposed) {
        setState(() => recordingStatusMessage = '녹음 오류: $e');
      }
    }
  }

  // 마디 녹음 시작
  Future<void> _startMeasureRecording() async {
    if (_isDisposed || _recorder == null) return;

    try {
      print('🎙️ 마디 ${_currentMeasure + 1} 녹음 시작: ${DateTime.now()}');

      await _recorder!.startRecorder(
        toFile: _recordingPath,
        codec: fs.Codec.pcm16WAV,
        sampleRate: 16000,
        numChannels: 1,
        bitRate: 16000,
      );

      if (!_isDisposed) {
        setState(() {
          recordingStatusMessage =
              '녹음 중... (마디: ${_currentMeasure + 1}/$_totalMeasures)';
        });
      }
    } catch (e) {
      print('❌ 마디 녹음 시작 중 오류 발생: $e');
      if (!_isDisposed) {
        setState(() => recordingStatusMessage = '녹음 시작 실패: $e');
      }
    }
  }

  /// WebSocket을 통해 녹음 데이터를 서버로 전송
  Future<void> _sendRecordingData() async {
    try {
      // State가 이미 dispose된 경우 바로 return
      if (!mounted || _isDisposed) {
        print('❌ State가 dispose된 후 _sendRecordingData 호출됨!');
        return;
      }
      if (_stompClient == null) {
        print('❌ _stompClient가 null입니다!');
        return;
      }
      if (!_stompClient!.connected) {
        print('❌ WebSocket이 연결되지 않았습니다!');
        return;
      }

      try {
        final file = File(_recordingPath!);
        if (await file.exists()) {
          final base64String = base64Encode(await file.readAsBytes());
          final message = {
            'email': _userEmail,
            'message': base64String,
            'currentMeasure': _currentMeasure,
            'totalMeasures': _totalMeasures
          };

          _stompClient!.send(
            destination: '/app/audio/forwarding',
            body: json.encode(message),
            headers: {'content-type': 'application/json'},
          );

          if (!_isDisposed) {
            setState(() => recordingStatusMessage =
                '녹음 데이터 전송 완료 (마디: ${_currentMeasure + 1}/$_totalMeasures)');
          }
        } else {
          print('⚠️ 녹음 파일이 존재하지 않습니다: $_recordingPath');
        }
      } catch (e) {
        print('❌ 녹음 데이터 전송 중 오류 발생: $e');
      }
    } catch (e, stack) {
      print('❌ _sendRecordingData 예외 발생: $e\n$stack');
    }
  }

  @override
  void dispose() {
    _isDisposed = true;

    // 리소스 정리
    cleanupResources();

    // 애니메이션 컨트롤러 해제
    _overlayController.dispose();

    widget.playbackController.onMeasureChange = null; // 이벤트 구독 해제
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    // 이 위젯은 추상적이므로 자체 UI가 없음
    return const SizedBox.shrink();
  }
}
