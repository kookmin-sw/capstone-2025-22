import 'dart:io';
import 'dart:ui';
import 'dart:convert';
import 'package:flutter_screenutil/flutter_screenutil.dart';
import 'package:capstone_2025/services/api_func.dart';
import 'package:capstone_2025/services/storage_service.dart';
import 'package:capstone_2025/widgets/linedText.dart';
import 'package:capstone_2025/widgets/openSheetModal.dart';
import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:font_awesome_flutter/font_awesome_flutter.dart';
import 'package:path_provider/path_provider.dart';

class MusicsheetDetail extends StatefulWidget {
  final String songID; // 노래 ID
  final String songTitle; // 노래 제목

  const MusicsheetDetail(
      {super.key, required this.songID, required this.songTitle});

  @override
  State<MusicsheetDetail> createState() => _MusicsheetDetailState();
}

class _MusicsheetDetailState extends State<MusicsheetDetail> {
  // 차트용 (날짜·점수)
  List<Map<String, String>> scoreData = [];
  // 테이블 + 디테일 호출용 practiceId 포함
  List<Map<String, dynamic>> practiceList = [];
  Uint8List? previewBytes; // 악보 프리뷰 이미지
  int? _selectedPracticeId; // 선택된 연습 기록 ID
  String? _xmlDataString; // 악보 XML 저장용

  @override
  void initState() {
    super.initState();
    _loadPreview();
    createDetailList();
    _fetchSheetXml();
  }

  // 로컬에 저장된 sheetInfo의 프리뷰용 fullSheetImage 가져오기
  Future<void> _loadPreview() async {
    final dir = await getApplicationDocumentsDirectory();
    final path = '${dir.path}/sheet_preview_${widget.songID}.png';
    final file = File(path);
    if (await file.exists()) {
      final bytes = await file.readAsBytes();
      setState(() {
        previewBytes = bytes;
      });
    }
  }

  // 채점 결과 렌더링용 악보 xml 파일 가져오기
  Future<void> _fetchSheetXml() async {
    final resp = await getHTTP('/sheets/${widget.songID}', {});
    if (resp['errMessage'] == null) {
      setState(() {
        // 1) 서버가 준 Base64
        final base64Xml = resp['body']['sheetInfo'] as String;
        // 2) Base64 → bytes → UTF8 문자열
        final bytes = base64Decode(base64Xml);
        var xml = utf8.decode(bytes);
        // 3) XML 선언이 없으면 붙여주기
        if (!xml.startsWith('<?xml')) {
          xml = '<?xml version="1.0" encoding="UTF-8"?>\n' + xml;
        }
        setState(() {
          _xmlDataString = xml;
        });
      });
    } else {
      debugPrint('악보 XML 로딩 실패: ${resp['errMessage']}');
    }
  }

  // 특정 악보 연습의 세부 정보 반환
  Future<Map<String, dynamic>> fetchPracticeDetail(int practiceId) async {
    final resp = await getHTTP('/sheets/practices/$practiceId', {});
    if (resp['errMessage'] != null) {
      throw Exception('연습 기록 상세 로딩 실패: ${resp['errMessage']}');
    }
    return resp['body'] as Map<String, dynamic>;
  }

  void createDetailList() async {
    // 리스트 생성 함수
    String? email = await storage.read(key: "user_email");
    final response = await getHTTP(
      '/sheets/${widget.songID}/practices',
      {"pageSize": 100, "pageNumber": 0, "email": email},
    );

    if (response['errMessage'] == null) {
      List<dynamic> rawData = response['body'];
      print("res: $rawData");
      final newList = rawData.map<Map<String, dynamic>>((item) {
        final rawDate = DateTime.tryParse(item['createdDate'] ?? '');
        final formattedDate = rawDate != null
            ? "${rawDate.year.toString().padLeft(4, '0')}.${rawDate.month.toString().padLeft(2, '0')}.${rawDate.day.toString().padLeft(2, '0')}"
            : '';
        return {
          'practiceId': item['practiceId'],
          '연습 날짜': formattedDate,
          '점수': item['score'].toString(),
        };
      }).toList();
      setState(() {
        practiceList = newList;
        // 차트용 데이터
        scoreData = newList
            .map((e) => {
                  '연습 날짜': e['연습 날짜'] as String,
                  '점수': e['점수'] as String,
                })
            .toList();
        // 기본 선택 첫 번째로 ID 지정
        if (practiceList.isNotEmpty) {
          _selectedPracticeId = practiceList.first['practiceId'] as int;
        }
      });
    }
  }

  void _onRowTap(int practiceId) {
    setState(() {
      // 항상 해당 practiceId가 선택되도록
      _selectedPracticeId = practiceId;
    });
  }

  List<Map<String, String>> lastFive = []; // 그래프에 출력할 마지막 5개 데이터

  List<FlSpot> generateChartData() {
    // scoreData에서 마지막 5개 데이터만 가져오기
    lastFive = scoreData.length > 5
        ? scoreData.sublist(scoreData.length - 5)
        : List.from(scoreData);

    // 차트 데이터로 변환 (x축은 0부터 시작)
    return List.generate(
      lastFive.length,
      (index) {
        final score = double.tryParse(lastFive[index]['점수']!) ?? 0.0;
        return FlSpot(index.toDouble(), score); // x축은 0부터 시작
      },
    );
  }

  // 최소값 찾기(그래프 하단 여백 남기기 위해)
  double getMinY() {
    if (scoreData.isEmpty) return 0.0;

    final scores =
        scoreData.map((data) => double.tryParse(data['점수']!) ?? 0.0).toList();
    final minScore = scores.reduce((a, b) => a < b ? a : b); // 최소 점수
    return (minScore - 5).clamp(0.0, 100.0);
  }

  // 그래프 생성
  Widget _buildGraph() {
    return Container(
      decoration: BoxDecoration(
        color: Colors.white, // 배경색 추가
        borderRadius: BorderRadius.circular(9),
        boxShadow: [
          BoxShadow(
            // 그래프 그림자 추가
            color: Colors.black.withOpacity(0.2),
            blurRadius: 5,
            spreadRadius: 1.5,
            offset: Offset(2, 2),
          ),
        ],
      ),
      child: Padding(
        padding: EdgeInsets.all(15.h),
        child: LineChart(
          // LineChart 위젯 추가
          LineChartData(
              // 그래프 상하좌우 여백
              minX: -0.1.w,
              maxX: 1.55.w,
              minY: getMinY(), // 최소값 - 5점
              maxY: 120.h,

              // 배경색
              backgroundColor: Colors.grey.shade100,

              // 그래프 속성
              gridData: FlGridData(
                show: true, // 그리드 라인 제거
                drawVerticalLine: true, // 세로선 그리기
                drawHorizontalLine: true, // 가로선 그리기
                getDrawingHorizontalLine: (value) => FlLine(
                  // 가로선 스타일
                  color: Colors.grey.shade400,
                  strokeWidth: 0.8,
                  dashArray: [4, 4],
                ),
                getDrawingVerticalLine: (value) => FlLine(
                  // 세로선 스타일
                  color: Colors.grey.shade400,
                  strokeWidth: 0.8,
                  dashArray: [4, 4],
                ),
              ),
              titlesData: FlTitlesData(
                // 축 제목
                leftTitles: AxisTitles(
                  sideTitles: SideTitles(
                    // 왼쪽 축 제목
                    showTitles: true,
                    reservedSize: 30,
                    getTitlesWidget: (value, meta) {
                      if (value > 100) {
                        // 점수 태그 추가
                        return Container(); // 100점 초과 값 숨김
                      }
                      if (value == getMinY()) return Container(); // 최소값 숨김
                      return Text(
                        // 점수 표시
                        value.toInt().toString(),
                        style: TextStyle(
                            fontSize: 12, color: Colors.grey.shade700),
                      );
                    },
                    interval: 10, // 0~100 점수 기준
                  ),
                ),
                rightTitles: AxisTitles(
                  sideTitles: SideTitles(showTitles: false), // 오른쪽 축 제거
                ),
                topTitles: AxisTitles(
                  sideTitles: SideTitles(showTitles: false), // 위쪽 축 제거
                ),
                bottomTitles: AxisTitles(
                  sideTitles: SideTitles(showTitles: false), // 아래쪽 축 제거
                ),
              ),
              borderData: FlBorderData(
                show: true,
                border: Border(
                  left: BorderSide(color: Colors.black26, width: 2),
                  bottom: BorderSide(color: Colors.black26, width: 2),
                ),
              ),
              lineBarsData: [
                // 그래프 데이터
                LineChartBarData(
                  spots: generateChartData(),
                  isCurved: false,
                  color: Color(0xffD97D6C), // 그래프 색상
                  barWidth: 2,
                  isStrokeCapRound: true,
                ),
              ],
              lineTouchData: LineTouchData(
                // 점 클릭 시 툴팁
                touchTooltipData: LineTouchTooltipData(
                  getTooltipColor: (LineBarSpot spot) => Colors.white,
                  tooltipRoundedRadius: 12,
                  tooltipPadding:
                      EdgeInsets.symmetric(horizontal: 25, vertical: 5),
                  getTooltipItems: (List<LineBarSpot> touchedSpots) {
                    return touchedSpots.map((spot) {
                      // TextSpan으로 여러 스타일을 적용
                      return LineTooltipItem(
                        // 날짜와 점수 두 부분으로 나누기
                        '',
                        TextStyle(
                          color: Colors.black,
                          fontWeight: FontWeight.w500,
                          fontSize: 13.sp,
                        ),
                        // 여러 스타일을 적용할 수 있는 TextSpan 사용
                        textAlign: TextAlign.center,
                        children: [
                          TextSpan(
                            text: '${spot.y.toInt()}점',
                            style: TextStyle(
                              fontSize: 7.sp, // 점수는 크게
                              color: Color(0xffD97D6C), // 점수는 주황색
                              fontWeight: FontWeight.bold, // 점수는 볼드체
                            ),
                          ),
                          TextSpan(
                            text: '\n${lastFive[(spot.x).toInt()]['연습 날짜']}',
                            style: TextStyle(
                              fontSize: 4.sp, // 날짜는 작게
                              color: Colors.black, // 날짜는 검은색
                            ),
                          ),
                        ],
                      );
                    }).toList();
                  },
                ),
                touchCallback: // 터치 이벤트
                    (FlTouchEvent event, LineTouchResponse? response) {},
                handleBuiltInTouches: true,
              )),
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Theme.of(context).scaffoldBackgroundColor,
      body: Padding(
        padding: EdgeInsets.only(
          left: 8.w,
          right: 8.w,
          top: 20.h,
        ),
        child: Column(
          children: [
            SizedBox(
              // 뒤로가기 버튼
              width: double.infinity,
              child: Row(
                children: [
                  GestureDetector(
                    onTap: () => Navigator.pop(context),
                    child: FaIcon(
                      FontAwesomeIcons.chevronLeft,
                      size: 9.sp,
                      color: Color(0xff646464),
                    ),
                  ),
                  Expanded(
                    // 노래 제목
                    child: Center(
                      child: linedText(
                        widget.songTitle,
                        10.sp,
                        Colors.black54,
                        Colors.white,
                        4,
                      ),
                    ),
                  ),
                ],
              ),
            ),
            SizedBox(height: 7.h),
            Expanded(
              // 그래프와 표
              child: Padding(
                padding: EdgeInsets.symmetric(
                  horizontal: 5.w,
                ),
                child: IntrinsicHeight(
                  child: Row(
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: [
                      Expanded(
                        flex: 2,
                        child: Column(
                          children: [
                            SizedBox(height: 10.h),
                            Expanded(
                              // 그래프
                              flex: 5,
                              child: Container(
                                decoration: BoxDecoration(
                                  color: Colors.white,
                                  borderRadius: BorderRadius.circular(9),
                                  boxShadow: [
                                    BoxShadow(
                                      color: Colors.black.withOpacity(0.2),
                                      blurRadius: 5,
                                      spreadRadius: 1.5,
                                      offset: Offset(2, 2),
                                    ),
                                  ],
                                ),
                                child: _buildGraph(),
                              ),
                            ),
                            SizedBox(height: 10.h),
                            Expanded(
                              // 표
                              flex: 5,
                              child: Container(
                                decoration: BoxDecoration(
                                  color: Colors.white,
                                  borderRadius: BorderRadius.circular(9),
                                  boxShadow: [
                                    BoxShadow(
                                      color: Colors.black.withOpacity(0.2),
                                      blurRadius: 5,
                                      spreadRadius: 1.5,
                                      offset: Offset(2, 2),
                                    ),
                                  ],
                                ),
                                child: _buildScoreTable(),
                              ),
                            ),
                            SizedBox(height: 10.h),
                          ],
                        ),
                      ),
                      SizedBox(width: 4.w),
                      Expanded(
                        // 채점 결과 악보
                        flex: 3,
                        child: Padding(
                          padding: EdgeInsets.all(8.h),
                          child: Stack(
                            children: [
                              Container(
                                height: double.infinity,
                                width: double.infinity,
                                decoration: BoxDecoration(
                                  color: Colors.white,
                                  borderRadius: BorderRadius.circular(9),
                                  boxShadow: [
                                    BoxShadow(
                                      color: Colors.black.withOpacity(0.2),
                                      blurRadius: 3,
                                      spreadRadius: 1.5,
                                      offset: Offset(2, 2),
                                    ),
                                  ],
                                ),
                                child: ClipRRect(
                                  borderRadius: BorderRadius.circular(9),
                                  child: Stack(
                                    fit: StackFit.expand,
                                    children: [
                                      // 악보 프리뷰 이미지 - Todo : 이미지 안 떠서 사이즈 제대로 설정 못함
                                      if (previewBytes != null)
                                        Positioned(
                                          top: 20.h,
                                          left: 20.w,
                                          right: 20.w,
                                          bottom: 50.h,
                                          child: Image.memory(previewBytes!,
                                              fit: BoxFit.cover),
                                        )
                                      else
                                        Center(
                                            child: CircularProgressIndicator()),
                                      // 블러 오버레이
                                      Positioned.fill(
                                        child: BackdropFilter(
                                          filter: ImageFilter.blur(
                                              sigmaX: 2, sigmaY: 2),
                                          child: Container(
                                            color:
                                                Colors.white.withOpacity(0.3),
                                          ),
                                        ),
                                      ),
                                    ],
                                  ),
                                ),
                              ),
                              Positioned(
                                // 악보 확대 버튼
                                top: MediaQuery.of(context).size.height * 0.755,
                                right:
                                    MediaQuery.of(context).size.height * 0.013,
                                child: IconButton(
                                  onPressed: () async {
                                    if (_selectedPracticeId == null ||
                                        _xmlDataString == null) return;

                                    final BuildContext localContext = context;
                                    try {
                                      final detail = await fetchPracticeDetail(
                                          _selectedPracticeId!);
                                      if (!mounted) return;

                                      final rawInfo = detail['practiceInfo']
                                          as List<dynamic>;
                                      final practiceInfo = rawInfo
                                          .map((e) => Map<String, dynamic>.from(
                                              e as Map))
                                          .toList();

                                      openMusicSheet(
                                        context: localContext,
                                        xmlDataString: _xmlDataString!,
                                        practiceInfo: practiceInfo,
                                      );
                                    } catch (e) {
                                      if (!mounted) return;
                                      ScaffoldMessenger.of(localContext)
                                          .showSnackBar(
                                        SnackBar(content: Text('상세 로딩 실패: $e')),
                                      );
                                    }
                                  },
                                  icon: FaIcon(FontAwesomeIcons.expand,
                                      size: 9.sp, color: Color(0xffD97D6C)),
                                ),
                              ),
                            ],
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildScoreTable() {
    return Stack(
      // 표와 헤더 겹치기 - 헤더 두께 조정을 위해
      children: [
        Positioned.fill(
          top: 20.h,
          child: Scrollbar(
            thumbVisibility: true, // 항상 스크롤바 보이기
            thickness: 8, // 스크롤바 두께 조정
            radius: Radius.circular(10), // 스크롤바 끝부분 둥글게 처리
            child: ListView.builder(
              itemCount: practiceList.length,
              padding: EdgeInsets.only(top: 13.h),
              physics: ClampingScrollPhysics(),
              shrinkWrap: true,
              itemBuilder: (context, index) {
                final item = practiceList[index];
                final isSelected = item['practiceId'] == _selectedPracticeId;

                return GestureDetector(
                  onTap: () {
                    setState(() {
                      _selectedPracticeId = item['practiceId'] as int;
                    });
                    _onRowTap(_selectedPracticeId!);
                  },
                  child: Container(
                      padding: EdgeInsets.symmetric(vertical: 15.h),
                      decoration: BoxDecoration(
                        color: isSelected
                            ? Colors.grey.shade100 // 선택된 색
                            : Colors.white,
                        border: Border(
                          bottom:
                              BorderSide(color: Colors.grey.shade300, width: 1),
                        ),
                      ),
                      child: Row(
                        children: [
                          _buildListCell(item["연습 날짜"]!, flex: 1),
                          _buildListCell(item["점수"]!, flex: 1, isCenter: true),
                        ],
                      )),
                );
              },
            ),
          ),
        ),
        Container(
          // 표 헤더
          height: 35.h,
          decoration: BoxDecoration(
            color: Color(0xffD97D6C),
            borderRadius: BorderRadius.vertical(top: Radius.circular(10)),
          ),
          child: Row(
            children: [
              _buildListHeaderCell("연습 날짜", flex: 1),
              _buildListHeaderCell("점수", flex: 1, isCenter: true),
            ],
          ),
        ),
      ],
    );
  }

  // 표 헤더 셀 생성
  Widget _buildListHeaderCell(String text,
      {int flex = 1, bool isCenter = false}) {
    if (isCenter) {
      // 가운데 정렬
      return Expanded(
        flex: flex,
        child: Center(
          child: Text(
            text,
            style: TextStyle(
              fontWeight: FontWeight.bold,
              color: Colors.white,
              fontSize: 6.sp,
            ),
          ),
        ),
      );
    }
    return Expanded(
      // 왼쪽 정렬
      flex: flex,
      child: Padding(
        padding: EdgeInsets.only(left: 23.w),
        child: Text(
          text,
          style: TextStyle(
              fontWeight: FontWeight.bold, color: Colors.white, fontSize: 6.sp),
          textAlign: TextAlign.start,
        ),
      ),
    );
  }

  // 표 셀 생성
  Widget _buildListCell(String text, {int flex = 1, bool isCenter = false}) {
    if (isCenter) {
      // 가운데 정렬
      return Expanded(
        flex: flex,
        child: Center(
          child: Text(
            text,
            style: TextStyle(fontSize: 7.sp, fontWeight: FontWeight.w500),
            textAlign: TextAlign.center,
          ),
        ),
      );
    }
    return Expanded(
      // 왼쪽 정렬
      flex: flex,
      child: Padding(
        padding: EdgeInsets.only(left: 19.w),
        child: Text(
          text,
          style: TextStyle(fontSize: 6.5.sp, fontWeight: FontWeight.w400),
          textAlign: TextAlign.left,
        ),
      ),
    );
  }
}
