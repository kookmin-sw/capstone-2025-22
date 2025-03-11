import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';
import 'package:font_awesome_flutter/font_awesome_flutter.dart';

class MusicsheetDetail extends StatefulWidget {
  const MusicsheetDetail({super.key});

  @override
  State<MusicsheetDetail> createState() => _MusicsheetDetailState();
}

class _MusicsheetDetailState extends State<MusicsheetDetail> {
  List<Map<String, String>> scoreData = [
    {'연습 날짜': "2025.01.18", '점수': "100"},
    {'연습 날짜': "2025.02.10", '점수': "95"},
    {'연습 날짜': "2025.02.11", '점수': "90"},
    {'연습 날짜': "2025.02.12", '점수': "85"},
    {'연습 날짜': "2025.02.17", '점수': "80"},
  ];

  List<FlSpot> generateChartData() {
    return scoreData.asMap().entries.map((entry) {
      final index = entry.key.toDouble();
      final score = double.tryParse(entry.value['점수']!) ?? 0.0;
      return FlSpot(index, score);
    }).toList();
  }

  Widget _buildGraph() {
    return Container(
      decoration: BoxDecoration(
        color: Colors.grey.shade200, // 배경색 추가
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
      child: Padding(
        padding: const EdgeInsets.all(8.0),
        child: LineChart(
          LineChartData(
            backgroundColor: Colors.grey.shade200,
            gridData: FlGridData(
              show: false,
              drawVerticalLine: true,
              drawHorizontalLine: true,
              getDrawingHorizontalLine: (value) => FlLine(
                color: Colors.grey.shade400,
                strokeWidth: 0.8,
                dashArray: [4, 4],
              ),
              getDrawingVerticalLine: (value) => FlLine(
                color: Colors.grey.shade400,
                strokeWidth: 0.8,
                dashArray: [4, 4],
              ),
            ),
            titlesData: FlTitlesData(
              leftTitles: AxisTitles(
                sideTitles: SideTitles(
                  showTitles: true,
                  reservedSize: 30,
                  getTitlesWidget: (value, meta) => Text(
                    value.toInt().toString(),
                    style: TextStyle(fontSize: 12, color: Colors.grey.shade700),
                  ),
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
            borderData: FlBorderData(show: true),
            lineBarsData: [
              LineChartBarData(
                spots: generateChartData(),
                isCurved: true,
                gradient: LinearGradient(
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
          ),
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
                    child: Center(
                      child: Text(
                        "악보 1",
                        style: TextStyle(
                          fontSize: 25,
                          fontWeight: FontWeight.w900,
                        ),
                        textAlign: TextAlign.center,
                      ),
                    ),
                  ),
                ],
              ),
            ),
            SizedBox(height: 7),
            Expanded(
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
                      flex: 3,
                      child: Padding(
                        padding: const EdgeInsets.all(8.0),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.end,
                          children: [
                            Expanded(
                              flex: 20,
                              child: Container(
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
      children: [
        Positioned.fill(
          top: 20,
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
        Container(
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

  Widget _buildListHeaderCell(String text,
      {int flex = 1, bool isCenter = false}) {
    if (isCenter) {
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

  Widget _buildListCell(String text, {int flex = 1, bool isCenter = false}) {
    if (isCenter) {
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
