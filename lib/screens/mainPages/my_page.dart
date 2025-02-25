import 'package:flutter/material.dart';
import 'package:font_awesome_flutter/font_awesome_flutter.dart';

class MyPage extends StatefulWidget {
  const MyPage({super.key});

  @override
  State<MyPage> createState() => _MyPageState();
}

class _MyPageState extends State<MyPage> {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: Padding(
          padding: EdgeInsets.all(20),
          child: Column(
            children: [
              Container(
                // 프로필 정보
                padding: EdgeInsets.symmetric(
                  vertical: 10,
                  horizontal: 20,
                ),

                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(10),
                ),
                child: Row(
                  // 프로필 정보 내용
                  children: [
                    Padding(
                      padding: const EdgeInsets.all(8.0),
                      child: SizedBox(
                        height: 70,
                        width: 70,
                        child: CircleAvatar(
                          // 프로필 이미지
                          radius: 40,
                          backgroundColor: Colors.grey[300], // 배경색 지정
                          child: Icon(
                            Icons.person,
                            size: 60,
                            color: Colors.white,
                          ), // 기본 아이콘
                        ),
                      ),
                    ),
                    SizedBox(
                      width: 20,
                    ),
                    Expanded(
                      child: Column(
                        // 이름 및 이메일
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Row(
                            children: [
                              Text(
                                "홍길동",
                                style: TextStyle(
                                  fontSize: 23,
                                  fontWeight: FontWeight.bold,
                                ),
                              ),
                              SizedBox(
                                width: 20,
                              ),
                              FaIcon(FontAwesomeIcons.edit, size: 22),
                            ],
                          ),
                          Text(
                            "example@gamil.com",
                            style: TextStyle(fontSize: 19, color: Colors.grey),
                          )
                        ],
                      ),
                    ),
                  ],
                ),
              ),
              SizedBox(
                height: 15,
              ),
              Padding(
                padding: const EdgeInsets.only(left: 20),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Icon(
                      Icons.queue_music_rounded,
                      size: 32,
                      color: Color(0xff646464),
                    ),
                    SizedBox(
                      width: 7,
                    ),
                    Text(
                      "악보 연습 기록",
                      style: TextStyle(
                        fontSize: 20,
                        fontWeight: FontWeight.bold,
                        color: Color(0xff646464),
                      ),
                    )
                  ],
                ),
              ),
              Spacer(
                flex: 1,
              ),
              Expanded(
                flex: 3,
                child: Text(
                  "지정된 악보가 없습니다. \n악보 연습에서 악보를 추가해보세요!",
                  textAlign: TextAlign.center,
                  style: TextStyle(
                      fontSize: 22,
                      color: Colors.grey.withOpacity(0.7),
                      fontWeight: FontWeight.bold),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
