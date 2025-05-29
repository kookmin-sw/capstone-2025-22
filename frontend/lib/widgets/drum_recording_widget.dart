// ignore_for_file: avoid_print
import 'dart:io';
import 'dart:async';
import 'dart:convert';
import 'package:xml/xml.dart';
import 'package:logger/logger.dart';
import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';
import 'package:flutter_sound/flutter_sound.dart' as fs;
import 'package:android_intent_plus/android_intent.dart';
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

  /// MusicXML 파일 내용
  final String? xmlDataString;

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

  final Future<String?> Function() fetchPracticeIdentifier;

  /// 채점 결과 콜백
  final void Function(Map<String, dynamic> gradingResult)? onGradingResult;
  final int? userSheetId;
  final int? patternId;

  const DrumRecordingWidget({
    super.key,
    this.userSheetId,
    this.patternId,
    required this.title,
    required this.xmlDataString,
    required this.audioFilePath,
    required this.playbackController,
    required this.fetchPracticeIdentifier, // identifier 가져옴
    this.onRecordingComplete,
    this.onMeasureUpdate,
    this.onOnsetsReceived,
    this.onMusicXMLParsed,
    this.onGradingResult,
  });

  @override
  State<DrumRecordingWidget> createState() => DrumRecordingWidgetState();
}

class DrumRecordingWidgetState extends State<DrumRecordingWidget>
    with SingleTickerProviderStateMixin {
// 상태 플래그
  bool _isRecorderReady = false;
  bool _isDisposed = false;
  bool isRecording = false;
  bool isCountingDown = false;
  bool _webSocketConnected = false;

  // WebSocket 관련
  StompClient? _stompClient;
  int _reconnectAttemps = 0;
  final int _maxReconnectAttempts = 5;
  String _userEmail = '';
  final _storage = const FlutterSecureStorage();
  Function? _stompUnsubscribe;
  String? _identifier;

  // 녹음 관련
  String? _recordingPath;
  fs.FlutterSoundRecorder? _recorder;
  String recordingStatusMessage = '';
  Timer? _recordingTimer;
  StreamSubscription<fs.RecordingDisposition>? _recorderSubscription;

  // 온셋 감지 및 딜레이 측정을 위한 변수
  bool firstBufferReceived = false; // 첫 오디오 버퍼 수신 여부
  DateTime? firstBufferTime; // 첫 오디오 버퍼 수신 시각
  DateTime? recordingStartTime; // 녹음 시작 시각

  // XML 파싱 및 타이밍 관련
  int _beatsPerMeasure = 4;
  int _beatType = 4;
  int _totalMeasures = 1;
  double _totalDuration = 0.0;
  int _currentMeasure = 0;
  double _secondsPerMeasure = 0.0; // 한 마디당 시간(초), XML 파싱 후 계산됨

  // 카운트다운 관련
  int countdown = 3;
  Timer? _countdownTimer;
  late AnimationController _overlayController;
  late Animation<double> _overlayAnimation;

  // 결과
  List<dynamic> _detectedOnsets = [];
  int _receivedResults = 0; // answerOnsetPlayed 메시지 수 카운트

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

    _initializeAll().then((_) {
      _isRecorderReady = true;
      print('녹음기 준비됨');

      // 녹음기가 준비되면 오디오 데이터 수신 리스너 등록
      if (_recorder != null) {
        _setupAudioDataListener();
      }
    });

    // PlaybackController의 이벤트 구독
    widget.playbackController.onMeasureChange = _handleMeasureChange;
    widget.playbackController.onCountdownComplete = _handleCountdownComplete;
    widget.playbackController.onPlaybackComplete = _handlePlaybackComplete;
  }

  // 오디오 데이터 수신 리스너 등록 함수
  void _setupAudioDataListener() {
    firstBufferReceived = false;

    _recorderSubscription?.cancel();
    _recorderSubscription = _recorder!.onProgress!.listen((event) {
      // fs.RecordingDisposition 이벤트 예시 사용
      // 실제 버퍼 데이터를 받는 스트림이 있다면 그걸 사용해야 함
      final now = DateTime.now();

      if (!firstBufferReceived) {
        firstBufferReceived = true;
        firstBufferTime = now;
        if (recordingStartTime != null) {
          final bufferDelay =
              now.difference(recordingStartTime!).inMilliseconds / 1000.0;
          print("첫 버퍼 수신까지 지연 시간: $bufferDelay 초");
        }
      }

      // 임시로 event.duration 을 온셋으로 사용 (실제 버퍼 분석 로직 대체 필요)
      Duration onsetDuration = detectOnset(event, now);
      print("감지된 온셋 타임스탬프: $onsetDuration, 현재 시각: $now");

      if (recordingStartTime != null) {
        final relativeOnset = onsetDuration.inMilliseconds / 1000.0;
        print("상대 온셋 시간: $relativeOnset 초");

        if (_webSocketConnected) {
          final sendStart = DateTime.now();
          _stompClient?.send(
            destination: (widget.patternId != null)
                ? '/app/pattern'
                : '/app/audio/forwarding',
            body: jsonEncode({'onset': relativeOnset}),
            headers: {
              'content-type': 'application/json',
            },
          );
          final sendEnd = DateTime.now();

          final sendDuration = sendEnd.difference(sendStart).inMilliseconds;
          print("웹소켓 전송 완료, 소요 시간: ${sendDuration}ms, 전송 시각: $sendEnd");
        } else {
          print('❌ 웹소켓 연결 안 됨, 전송 불가');
        }
      }
    });
  }

  /// 간단 샘플 온셋 감지 함수 (실제 신호 분석 로직 대체 필요)
  Duration detectOnset(fs.RecordingDisposition event, DateTime now) {
    if (recordingStartTime != null) {
      return now.difference(recordingStartTime!);
    } else {
      return Duration.zero;
    }
  }

  Future<void> _initializeAll() async {
    await _parseMusicXML(); // 1) XML 파싱 완료 보장
    await _initializeData(); // 2) Recorder·WebSocket 초기화
  }

  Future<void> openManageAllFilesSettings() async {
    if (Platform.isAndroid) {
      final intent = AndroidIntent(
        action: 'android.settings.MANAGE_APP_ALL_FILES_ACCESS_PERMISSION',
        data: 'package:com.example.capstone_2025',
      );
      await intent.launch();
    }
  }

  Future<void> _initializeData() async {
    print('[InitData] ▶️ 시작');
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

    // 1) 마이크 권한 요청
    var status = await Permission.microphone.request();
    if (status != PermissionStatus.granted) {
      throw RecordingPermissionException('마이크 권한이 부여되지 않았습니다.');
    }
    print('[Permission] ✅ 마이크 권한 획득');

    // 2) 기존 레코더가 있으면 닫기
    if (_recorder != null) {
      try {
        await _recorder!.closeRecorder();
      } catch (e) {
        print('⚠️ 기존 레코더 종료 중 오류: $e');
      }
    }

    // 녹음기 초기화
    _recorder = fs.FlutterSoundRecorder(logLevel: Level.off);
    await _recorder?.openRecorder();
    print('🎤 녹음기 초기화 완료');

    // 녹음 파일 저장 경로 설정
    final dir = await getApplicationDocumentsDirectory();
    _recordingPath = '${dir.path}/current_measure.aac';

    _isRecorderReady = true;
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
          // _subscribeToTopic();
        },
        beforeConnect: () async => print('🌐 WebSocket 연결 시도 중...'),
        onWebSocketError: (dynamic error) {
          print('❌ WebSocket 오류 발생: $error');
          if (!_isDisposed) {
            _retryWebSocketConnect();
          }
        },
        // STOMP 계층에서 에러가 왔을 때
        onStompError: (StompFrame frame) {
          print('❌ STOMP 프로토콜 에러: ${frame.body}');
        },
        // 핸들링되지 않은 모든 프레임을 찍어본다
        onUnhandledFrame: (dynamic frame) {
          print('⚠️ Unhandled STOMP frame: $frame');
        },
        onUnhandledMessage: (StompFrame frame) {
          print('⚠️ Unhandled STOMP message: ${frame.body}');
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

  // 구독
  void _subscribeToTopic() {
    if (_isDisposed || _stompClient == null) return;

    _stompUnsubscribe = _stompClient!.subscribe(
      destination: '/topic/onset/$_userEmail/$_identifier', // 구독 경로
      callback: (frame) {
        print(
            '[WebSocket 데이터 수신 완료] headers=${frame.headers}, body=${frame.body}');
        if (_isDisposed) return;

        if (frame.body != null) {
          final response = json.decode(frame.body!);

          if (response.containsKey('onsets')) {
            if (!_isDisposed) {
              setState(() {
                _detectedOnsets = response['onsets'];
              });
            }

            // 부모 위젯에 콜백으로 알림
            if (widget.onOnsetsReceived != null && !_isDisposed) {
              widget.onOnsetsReceived!(_detectedOnsets);
            }
          }
          // ② answerOnsetPlayed → 채점 결과
          if (response.containsKey('answerOnsetPlayed')) {
            _receivedResults++;
            widget.onGradingResult?.call(response);
            // 모든 마디 채점 결과 받았으면 녹음 종료
            if (_receivedResults >= _totalMeasures && isRecording) {
              stopRecording();
              widget.onRecordingComplete?.call(_detectedOnsets);
            }
          } else {
            print('⚠️ 빈 WebSocket 프레임 수신');
          }
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

  Future<void> _parseMusicXML() async {
    if (_isDisposed || widget.xmlDataString == null) return;

    try {
      String xmlDataString = widget.xmlDataString!;

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

  // PlaybackController 콜백 메소드 - 카운트다운 완료 처리
  void _handleCountdownComplete() async {
    print('[Countdown] ▶️ 완료 콜백 진입');
    print('[Countdown] recorder 상태: isStopped=${_recorder?.isStopped}');
    await _initRecorder();
    print('[Countdown] ✅ _initRecorder() 리턴');

    if (!mounted || _isDisposed) return;

    print('[Countdown] ▶️ 완료 콜백 진입 (_isRecorderReady=$_isRecorderReady)');
    if (!_isRecorderReady) {
      print('[Countdown] ❌ recorder not ready, skip startRecording');
      return;
    }
    // 녹음기가 아직 열려있지 않으면 재초기화
    if (_recorder == null || !(_recorder!.isStopped ?? false)) {
      await _initRecorder();
    }

    startRecording();
    print('[Countdown] ▶️ startRecording 호출됨');
  }

  // 연주 완료 처리
  void _handlePlaybackComplete(int lastMeasure) {
    if (!isRecording || _isDisposed) return;
    print('🎼 연주 완료 감지: 마지막 마디 $lastMeasure');

    if (_recorder!.isRecording) {
      _recorder!.stopRecorder().then((_) {
        print('🎙️ 마지막 마디 녹음 중지 완료');
        return _sendRecordingData();
      }).then((_) {
        print('📤 마지막 마디 녹음 데이터 전송 완료: ${DateTime.now()}');
        return stopRecording();
      }).catchError((error) {
        print('❌ 마지막 마디 처리 중 오류 발생: $error');
      });
    } else {
      stopRecording();
    }
  }

// 마디 변경 처리
  void _handleMeasureChange(int measureNumber) {
    if (!isRecording || _isDisposed) return;

    // [디버깅용] 마디 변경 감지 시각 출력
    print('🎼 마디 변경 감지: ${_currentMeasure + 1} -> ${measureNumber + 1} '
        'at ${DateTime.now().toIso8601String()}');
    // 첫 번째 마디 변경 감지인 경우 (녹음 시작)
    if (_currentMeasure == 0 && measureNumber == 0) {
      _startMeasureRecording();
      return;
    }

    // 측정값 측정이 변경될 때만 현재 마디 처리
    if (measureNumber > _currentMeasure) {
      // 현재 마디 녹음 중지 및 데이터 전송
      _processCurrentMeasure();
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

  /// 오디오 녹음 시작 시각 기록
  void _recordingStarted() {
    recordingStartTime = DateTime.now();
    firstBufferReceived = false;
    print("녹음 시작 시각: $recordingStartTime");
  }

  /// 오디오 녹음 시작
  void startRecording() async {
    if (isRecording || !mounted || _isDisposed || _recorder == null) return;

    _receivedResults = 0;
    _detectedOnsets.clear();

    // WebSocket 연결 확인
    if (!_webSocketConnected) {
      print('❌ 녹음을 시작할 수 없습니다: WebSocket이 연결되지 않았습니다.');
      setState(() => recordingStatusMessage = 'WebSocket 연결이 필요합니다!');
      return;
    }

    // 한 번만 fetch
    _identifier = await widget.fetchPracticeIdentifier();
    if (_identifier == null) {
      setState(() => recordingStatusMessage = '식별자 획득 실패');
      return;
    }

    // 식별자 획득 후 구독
    _subscribeToTopic();

    try {
      // 전체 녹음 프로세스 시작
      setState(() {
        isRecording = true;
        _currentMeasure = 0;
        recordingStatusMessage = '녹음이 시작되었습니다.';
      });

      _recordingStarted(); // 녹음 시작 시각 기록

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
    print('▶ stopRecording 호출됨 at ${DateTime.now().toIso8601String()}');

    if (!isRecording || !mounted || _isDisposed || _recorder == null) return;

    try {
      if (_recorder!.isRecording) {
        await _recorder!.stopRecorder(); // 현재 진행 중인 녹음 중지
        print('🎙️ 전체 녹음 중지 완료 시각: ${DateTime.now().toIso8601String()}');
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

  /// 녹음 일시정지
  Future<void> pauseRecording() async {
    if (!isRecording) return;
    await _recorder?.pauseRecorder();
  }

  /// 녹음 재개
  Future<void> resumeRecording() async {
    if (!isRecording) return;

    try {
      // 녹음기가 null인 경우 초기화
      if (_recorder == null) {
        await _initRecorder();
      }

      await _recorder?.resumeRecorder();
      print('▶️ 녹음 재개 (마디 ${_currentMeasure + 1}부터)');
    } catch (e) {
      print('❌ 녹음 재개 중 오류 발생: $e');
      // 오류 발생 시 녹음기 재초기화 시도
      try {
        await _initRecorder();
        await _recorder?.resumeRecorder();
      } catch (retryError) {
        print('❌ 녹음기 재초기화 및 재개 실패: $retryError');
      }
    }
  }

  // 마디 단위 처리
  Future<void> _processCurrentMeasure() async {
    if (!isRecording || _isDisposed || _recorder == null) return;

    try {
      if (_recorder!.isRecording) {
        await _recorder!.stopRecorder();
        final stopTime = DateTime.now();
        print('🎙️ 마디 ${_currentMeasure + 1} 녹음 중지 시각: $stopTime');
      }

      await _sendRecordingData();
      print('📤 마디 ${_currentMeasure + 1} 녹음 데이터 전송 완료: ${DateTime.now()}');
      widget.onMeasureUpdate?.call(_currentMeasure + 1, _totalMeasures);

      if (_currentMeasure < _totalMeasures - 1) {
        _currentMeasure++;
        if (_currentMeasure > 0) {
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
      widget.onMeasureUpdate?.call(_currentMeasure + 1, _totalMeasures);

      await _recorder!.startRecorder(
        toFile: _recordingPath,
        codec: fs.Codec.aacADTS,
        sampleRate: 16000,
        numChannels: 1,
        bitRate: 16000,
      );

      // 녹음 실제 시작 시각 기록 — startRecorder가 완료된 바로 직후!
      recordingStartTime = DateTime.now();
      firstBufferReceived = false;
      print("🎙️ 마디 ${_currentMeasure + 1} 실제 녹음 시작 시각: $recordingStartTime");

      widget.onMeasureUpdate?.call(_currentMeasure + 1, _totalMeasures);

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

  // WebSocket을 통해 녹음 데이터를 서버로 전송
  Future<void> _sendRecordingData() async {
    if (!mounted ||
        _isDisposed ||
        _stompClient == null ||
        !_stompClient!.connected) {
      print('❌ 녹음 데이터 전송 불가: 연결 상태 확인 필요');
      return;
    }

    // 전송 시작 시각
    print('📤 [${DateTime.now().toIso8601String()}] 녹음 데이터 전송 시작 '
        '(마디: ${_currentMeasure + 1})');

    try {
      final file = File(_recordingPath!);
      if (await file.exists()) {
        final base64String = base64Encode(await file.readAsBytes());
        print('📁 녹음 파일 크기: ${base64String.length} bytes');
        // print(base64String);

        final originalBpm =
            ((_beatsPerMeasure * 60) / (_totalDuration / _totalMeasures))
                .toInt();
        final adjustedBpm =
            (originalBpm * widget.playbackController.speed).round();

        final Map<String, dynamic> payload = (widget.patternId != null)
            // 패턴 필인 페이지
            ? {
                'bpm': adjustedBpm,
                'patternId': widget.patternId,
                'identifier': _identifier,
                'email': _userEmail,
                'audioBase64': base64String,
                'measureNumber': (_currentMeasure + 1).toString(),
                'endOfMeasure': _currentMeasure + 1 == _totalMeasures,
              }
            // 악보 연습 페이지
            : {
                'bpm': adjustedBpm,
                'userSheetId': widget.userSheetId,
                'identifier': _identifier,
                'email': _userEmail,
                'message': base64String,
                'measureNumber': (_currentMeasure + 1).toString(),
                'endOfMeasure': _currentMeasure + 1 == _totalMeasures,
              };

        _stompClient!.send(
          destination: (widget.patternId != null)
              ? '/app/pattern' // 패턴 필인 페이지
              : '/app/audio/forwarding', // 악보 연습 페이지
          body: json.encode(payload),
          headers: {
            'content-type': 'application/json',
            'receipt': 'measure-${_currentMeasure + 1}',
          },
        );

        // 전송 완료 시각
        print('📤 [${DateTime.now().toIso8601String()}] 녹음 데이터 전송 완료 '
            '(마디: ${_currentMeasure + 1})');

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
  }

  /// 카운트다운 오버레이 위젯 반환
  Widget buildCountdownOverlay() {
    if (!isCountingDown) return const SizedBox.shrink();

    return FadeTransition(
      opacity: _overlayAnimation,
      child: Container(
        color: Colors.black.withValues(alpha: 0.6),
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

  /// 모든 리소스를 안전하게 정리하는 메서드
  void cleanupResources() async {
    print('🧹 리소스 정리 시작...');

    // 모든 타이머 취소
    _countdownTimer?.cancel();
    _recordingTimer?.cancel();

    // 구독 취소
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

    // 녹음 중지 및 녹음기 정리
    if (_recorder != null) {
      try {
        if (_recorder!.isRecording) {
          await _recorder!.stopRecorder();
          print('✅ 녹음 중지 완료');
        }
        await _recorder!.closeRecorder();
        print('✅ 녹음기 종료 완료');
      } catch (e) {
        print('⚠️ 녹음 중지 중 오류 발생: $e');
      }
    }

    // WebSocket 연결 종료
    if (_stompClient != null) {
      try {
        _stompClient!.deactivate();
        print('✅ WebSocket 종료 완료');
      } catch (e) {
        print('⚠️ WebSocket 종료 중 오류 발생: $e');
      }
      _stompClient = null;
    }

    print('✅ 모든 리소스 정리 완료');
  }

  @override
  void dispose() {
    _isDisposed = true;

    // 1) 녹음 중이면 즉시 중지
    if (_recorder != null && _recorder!.isRecording) {
      try {
        _recorder!.stopRecorder();
      } catch (_) {}
    }
    // 2) 녹음기 닫기
    try {
      _recorder?.closeRecorder();
    } catch (_) {}

    // 3) WebSocket 구독 해제 & 연결 종료
    _stompUnsubscribe?.call();
    try {
      _stompClient?.deactivate();
    } catch (_) {}

    // 4) 타이머 취소
    _countdownTimer?.cancel();
    _recordingTimer?.cancel();

    // 5) 스트림 구독 취소
    _recorderSubscription?.cancel();

    // 6) PlaybackController 콜백 해제
    widget.playbackController
      ..onMeasureChange = null
      ..onCountdownComplete = null
      ..onPlaybackComplete = null;

    // 7) 애니메이션 컨트롤러 해제
    _overlayController.dispose();

    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    // 이 위젯은 추상적이므로 자체 UI가 없음
    return const SizedBox.shrink();
  }

  /// 마디별 녹음 데이터 전송
  Future<void> sendMeasureData({
    required int measureNumber,
    required bool isLastMeasure,
  }) async {
    if (!isRecording ||
        _isDisposed ||
        _stompClient == null ||
        !_stompClient!.connected) {
      print('❌ 녹음 데이터 전송 불가: 연결 상태 확인 필요');
      return;
    }

    try {
      final file = File(_recordingPath!);
      if (await file.exists()) {
        final base64String = base64Encode(await file.readAsBytes());
        final originalBpm =
            ((_beatsPerMeasure * 60) / (_totalDuration / _totalMeasures))
                .toInt();
        final adjustedBpm =
            (originalBpm * widget.playbackController.speed).round();

        final message = {
          'bpm': adjustedBpm,
          if (widget.patternId != null) // 이 부분 잘 동작하는지 확인해보기
            'patternId': widget.patternId!
          else
            'userSheetId': widget.userSheetId,
          'identifier': _identifier,
          'email': _userEmail,
          'message': base64String,
          'measureNumber': measureNumber.toString(),
          'endOfMeasure': isLastMeasure,
        };

        _stompClient!.send(
          destination: '/app/audio/forwarding',
          body: json.encode(message),
          headers: {
            'content-type': 'application/json',
            'receipt': 'measure-$measureNumber',
          },
        );

        if (!_isDisposed) {
          setState(() => recordingStatusMessage =
              '녹음 데이터 전송 완료 (마디: $measureNumber/$_totalMeasures)');
        }
      } else {
        print('⚠️ 녹음 파일이 존재하지 않습니다: $_recordingPath');
      }
    } catch (e) {
      print('❌ 녹음 데이터 전송 중 오류 발생: $e');
    }
  }
}
