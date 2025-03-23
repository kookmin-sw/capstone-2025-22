import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:font_awesome_flutter/font_awesome_flutter.dart';

class MusicsheetDetail extends StatefulWidget {
  final String songTitle; // 추가: 노래 제목 받기

  const MusicsheetDetail({super.key, required this.songTitle});

  @override
  State<MusicsheetDetail> createState() => _MusicsheetDetailState();
}

class _MusicsheetDetailState extends State<MusicsheetDetail> {
  List<Map<String, String>> scoreData = [
    {'연습 날짜': "2025.01.18", '점수': "70"},
    {'연습 날짜': "2025.02.10", '점수': "95"},
    {'연습 날짜': "2025.02.11", '점수': "100"},
    {'연습 날짜': "2025.02.12", '점수': "85"},
    {'연습 날짜': "2025.02.17", '점수': "90"},
    {'연습 날짜': "2025.02.18", '점수': "75"},
    {'연습 날짜': "2025.02.20", '점수': "88"},
    {'연습 날짜': "2025.02.22", '점수': "93"},
    {'연습 날짜': "2025.02.25", '점수': "78"},
    {'연습 날짜': "2025.02.28", '점수': "85"},
    {'연습 날짜': "2025.03.02", '점수': "97"},
    {'연습 날짜': "2025.03.05", '점수': "92"},
    {'연습 날짜': "2025.03.08", '점수': "80"},
    {'연습 날짜': "2025.03.10", '점수': "76"},
    {'연습 날짜': "2025.03.12", '점수': "98"},
    {'연습 날짜': "2025.03.15", '점수': "100"},
  ];

  List<Map<String, String>> lastFive = []; // 그래프에 출력할 마지막 5개 데이터

  List<FlSpot> generateChartData() {
    // scoreData에서 마지막 5개 데이터만 가져오기
    lastFive = scoreData.sublist(scoreData.length - 5, scoreData.length);

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
    final scores = scoreData.map((data) => double.tryParse(data['점수']!) ?? 0.0);
    final minScore = scores.reduce((a, b) => a < b ? a : b); // 최소 점수 찾기
    return (minScore - 5).clamp(0.0, 100.0); // 최소값에서 5점 감소 (최소 0점)
  }

  // 그래프 생성
  Widget _buildGraph() {
    return Container(
      decoration: BoxDecoration(
        color: Colors.grey.shade200, // 배경색 추가
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
        padding: const EdgeInsets.all(15),
        child: LineChart(
          // LineChart 위젯 추가
          LineChartData(
              // 그래프 상하좌우 여백
              minX: -0.3,
              maxX: 4.3,
              minY: getMinY(), // 최소값 - 5점
              maxY: 105,

              // 배경색
              backgroundColor: Colors.grey.shade200,

              // 그래프 속성
              gridData: FlGridData(
                show: false, // 그리드 라인 제거
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
                        return Padding(
                          padding: const EdgeInsets.only(bottom: 5),
                          child: Text(
                            "점수",
                            style: TextStyle(
                                fontSize: 10,
                                fontWeight: FontWeight.bold,
                                color: const Color.fromARGB(255, 46, 45, 45)),
                          ),
                        ); // 100점 초과 값 숨김
                      }
                      if (value == getMinY()) return Container(); // 최소값 숨김
                      return Text(
                        // 점수 표시
                        value.toInt().toString(),
                        style: TextStyle(
                            fontSize: 12, color: Colors.grey.shade700),
                      );
                    },
                    interval: 5, // 0~100 점수 기준
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
              borderData: FlBorderData(show: true),
              lineBarsData: [
                // 그래프 데이터
                LineChartBarData(
                  spots: generateChartData(),
                  isCurved: false,
                  gradient: LinearGradient(
                    // 그래프 그라데이션
                    colors: [Colors.orange.shade700, Colors.orange.shade400],
                  ),
                  barWidth: 3,
                  isStrokeCapRound: true,
                  belowBarData: BarAreaData(
                    show: true,
                    gradient: LinearGradient(
                      colors: [
                        Colors.orange.withOpacity(0.2),
                        Colors.transparent
                      ],
                    ),
                  ),
                ),
              ],
              lineTouchData: LineTouchData(
                // 점 클릭 시 툴팁
                touchTooltipData: LineTouchTooltipData(
                  getTooltipColor: (LineBarSpot spot) =>
                      Colors.amberAccent.withOpacity(0.5),
                  getTooltipItems: (List<LineBarSpot> touchedSpots) {
                    return touchedSpots.map((spot) {
                      return LineTooltipItem(
                        // 상세 정보 - 마지막 5개 데이터
                        '${lastFive[(spot.x).toInt()]['연습 날짜']}\n점수: ${spot.y.toInt()}점',
                        const TextStyle(
                          color: Colors.black,
                          fontWeight: FontWeight.w500,
                          fontSize: 10,
                        ),
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
        padding: const EdgeInsets.only(
          left: 20,
          right: 20,
          top: 10,
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
                      size: 25,
                    ),
                  ),
                  Expanded(
                    // 노래 제목
                    child: Center(
                        child: Stack(
                      children: [
                        // 외곽선 텍스트
                        Text(
                          widget.songTitle,
                          style: TextStyle(
                            fontSize: 27,
                            fontWeight: FontWeight.w900,
                            foreground: Paint()
                              ..style = PaintingStyle.stroke
                              ..strokeWidth = 4 // 외곽선 두께
                              ..color = Colors.black54, // 외곽선 색상
                          ),
                        ),
                        // 내부 색상 텍스트
                        Text(
                          widget.songTitle,
                          style: const TextStyle(
                            fontSize: 27,
                            fontWeight: FontWeight.w900,
                            color: Colors.white, // 내부 색상
                          ),
                        ),
                      ],
                    )),
                  ),
                ],
              ),
            ),
            SizedBox(height: 7),
            Expanded(
              // 그래프와 표
              child: Padding(
                padding: const EdgeInsets.symmetric(
                  horizontal: 25,
                ),
                child: Row(
                  children: [
                    Expanded(
                      flex: 2,
                      child: Column(
                        children: [
                          SizedBox(height: 10),
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
                          SizedBox(height: 10),
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
                          SizedBox(height: 10),
                        ],
                      ),
                    ),
                    SizedBox(width: 5),
                    Expanded(
                      // 악보 출력 예정 공간
                      flex: 3,
                      child: Padding(
                        padding: const EdgeInsets.all(8.0),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.end,
                          children: [
                            Expanded(
                              flex: 20,
                              child: Container(
                                child: Image.asset(
                                  'assets/images/image.png',
                                ),
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
          top: 20,
          child: Scrollbar(
            thumbVisibility: true, // 항상 스크롤바 보이기
            thickness: 8, // 스크롤바 두께 조정
            radius: Radius.circular(10), // 스크롤바 끝부분 둥글게 처리
            child: ListView.builder(
              itemCount: scoreData.length,
              padding: EdgeInsets.only(top: 10),
              physics: ClampingScrollPhysics(),
              shrinkWrap: true,
              itemBuilder: (context, index) {
                var item = scoreData[index];

                return Container(
                  padding: EdgeInsets.symmetric(vertical: 12),
                  decoration: BoxDecoration(
                    color: index.isEven ? Colors.white : Colors.grey.shade100,
                    border: Border(
                      bottom: BorderSide(color: Colors.grey.shade300, width: 1),
                    ),
                  ),
                  child: Row(
                    children: [
                      _buildListCell(item["연습 날짜"]!, flex: 1),
                      _buildListCell(item["점수"]!, flex: 1, isCenter: true),
                    ],
                  ),
                );
              },
            ),
          ),
        ),
        Container(
          // 표 헤더
          height: 30,
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
              fontSize: 16,
            ),
          ),
        ),
      );
    }
    return Expanded(
      // 왼쪽 정렬
      flex: flex,
      child: Padding(
        padding: const EdgeInsets.only(left: 65),
        child: Text(
          text,
          style: TextStyle(
              fontWeight: FontWeight.bold, color: Colors.white, fontSize: 16),
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
            style: TextStyle(fontSize: 19, fontWeight: FontWeight.w500),
            textAlign: TextAlign.center,
          ),
        ),
      );
    }
    return Expanded(
      // 왼쪽 정렬
      flex: flex,
      child: Padding(
        padding: const EdgeInsets.only(left: 50),
        child: Text(
          text,
          style: TextStyle(fontSize: 18, fontWeight: FontWeight.w400),
          textAlign: TextAlign.left,
        ),
      ),
    );
  }
}
