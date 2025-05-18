import 'package:flutter/material.dart';
import 'drum_info_popup.dart';
import 'package:flutter_screenutil/flutter_screenutil.dart';
import 'package:flutter_markdown/flutter_markdown.dart';

class DrumBasicsPage extends StatefulWidget {
  const DrumBasicsPage({super.key});

  @override
  State<DrumBasicsPage> createState() => _DrumBasicsPageState();
}

class _DrumBasicsPageState extends State<DrumBasicsPage> {
  final List<Map<String, String>> stepInfo = const [
    {
      'title': 'STEP 1  드럼 종류 알기',
      'subtitle': '드럼 북에는 어떤 종류가 있는지 알아보자!',
      'popupTitle': '드럼 종류',
      "imagePath": "assets/images/drum_kit.png",
      "description": """
**스네어 드럼 (Snare Drum)**  
드럼 세트의 중앙 앞쪽에 위치하며, 날카롭고 강한 소리를 냅니다. 대부분의 리듬에서 중심 박자를 담당하며, 가장 많이 사용되는 드럼입니다.

**베이스 드럼 (Bass Drum)**  
드럼 세트의 가장 아래쪽 중앙에 있는 큰 드럼으로, 발로 페달을 눌러 연주합니다. 둔탁하고 낮은 음을 내며 리듬의 기본이 되는 박자를 형성합니다.

**하이햇 (Hi-Hat)**  
좌측에 위치한 두 개의 심벌로 구성되며, 페달을 이용해 열고 닫을 수 있습니다. 닫힌 소리와 열린 소리를 조합해 다양한 리듬을 만들 수 있습니다.

**탐 1 (Tom 1)**  
스네어 드럼의 오른쪽 상단에 있는 비교적 작은 드럼입니다. 높은 음역의 필인(채움) 연주에 사용됩니다.

**탐 2 (Tom 2)**  
탐1보다 조금 더 큰 크기의 드럼으로, 탐1 오른쪽에 위치합니다. 중간 음역을 담당하며 탐1과 함께 사용됩니다.

**플로어 탐 (Floor Tom)**  
탐들 중 가장 크며, 바닥에 다리가 있는 구조입니다. 낮고 깊은 소리를 내며 주로 곡의 후반부나 마무리에 사용됩니다.

**크래시 심벌 (Crash Cymbal)**  
드럼 세트의 왼쪽 위쪽에 위치한 심벌로, 강하게 타격 시 날카롭고 폭발적인 소리를 냅니다. 주로 전환이나 강조 시 사용됩니다.

**라이드 심벌 (Ride Cymbal)**  
드럼 세트의 가장 오른쪽에 위치한 큰 심벌입니다. 일정한 리듬을 유지할 때 사용되며, 잔향이 긴 소리를 냅니다.
"""
    },
    {
      'title': 'STEP 2  드럼 용어 알기',
      'subtitle': '드럼 연주를 위해 필요한 용어를 알아보자!',
      'popupTitle': '드럼 용어 정리',
      "imagePath": '',
      "description":
          "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Nisl tincidunt eget nullam non. Quis hendrerit dolor magna eget est lorem ipsum dolor sit. Volutpat odio facilisis mauris sit amet massa. Commodo odio aenean sed adipiscing diam donec adipiscing tristique. Mi eget mauris pharetra et. Non tellus orci ac auctor augue. Elit at imperdiet dui accumsan sit. Ornare arcu dui vivamus arcu felis. Egestas integer eget aliquet nibh praesent. In hac habitasse platea dictumst quisque sagittis purus. Pulvinar elementum integer enim neque volutpat ac.Senectus et netus et malesuada. Nunc pulvinar sapien et ligula ullamcorper malesuada proin. Neque convallis a cras semper auctor. Libero id faucibus nisl tincidunt eget. Leo a diam sollicitudin tempor id. A lacus vestibulum sed arcu non odio euismod lacinia. In tellus integer feugiat scelerisque. Feugiat in fermentum posuere urna nec tincidunt praesent. Porttitor rhoncus dolor purus non enim praesent elementum facilisis. Nisi scelerisque eu ultrices vitae auctor eu augue ut lectus. Ipsum faucibus vitae aliquet nec ullamcorper sit amet risus. Et malesuada fames ac turpis egestas sed. Sit amet nisl suscipit adipiscing bibendum est ultricies. Arcu ac tortor dignissim convallis aenean et tortor at. Pretium viverra suspendisse potenti nullam ac tortor vitae purus. Eros donec ac odio tempor orci dapibus ultrices. Elementum nibh tellus molestie nunc. Et magnis dis parturient montes nascetur. Est placerat in egestas erat imperdiet. Consequat interdum varius sit amet mattis vulputate enim.Sit amet nulla facilisi morbi tempus. Nulla facilisi cras fermentum odio eu. Etiam erat velit scelerisque in dictum non consectetur a erat. Enim nulla aliquet porttitor lacus luctus accumsan tortor posuere. Ut sem nulla pharetra diam. Fames ac turpis egestas maecenas. Bibendum neque egestas congue quisque egestas diam. Laoreet id donec ultrices tincidunt arcu non sodales neque. Eget felis eget nunc lobortis mattis aliquam faucibus purus. Faucibus interdum posuere lorem ipsum dolor sit."
    },
    {
      'title': 'STEP 3  드럼 악보 알기',
      'subtitle': '드럼 악보를 어떻게 읽는지 알아보자!',
      'popupTitle': '드럼 악보 읽기',
      "imagePath": '',
      "description":
          "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Nisl tincidunt eget nullam non. Quis hendrerit dolor magna eget est lorem ipsum dolor sit. Volutpat odio facilisis mauris sit amet massa. Commodo odio aenean sed adipiscing diam donec adipiscing tristique. Mi eget mauris pharetra et. Non tellus orci ac auctor augue. Elit at imperdiet dui accumsan sit. Ornare arcu dui vivamus arcu felis. Egestas integer eget aliquet nibh praesent. In hac habitasse platea dictumst quisque sagittis purus. Pulvinar elementum integer enim neque volutpat ac.Senectus et netus et malesuada. Nunc pulvinar sapien et ligula ullamcorper malesuada proin. Neque convallis a cras semper auctor. Libero id faucibus nisl tincidunt eget. Leo a diam sollicitudin tempor id. A lacus vestibulum sed arcu non odio euismod lacinia. In tellus integer feugiat scelerisque. Feugiat in fermentum posuere urna nec tincidunt praesent. Porttitor rhoncus dolor purus non enim praesent elementum facilisis. Nisi scelerisque eu ultrices vitae auctor eu augue ut lectus. Ipsum faucibus vitae aliquet nec ullamcorper sit amet risus. Et malesuada fames ac turpis egestas sed. Sit amet nisl suscipit adipiscing bibendum est ultricies. Arcu ac tortor dignissim convallis aenean et tortor at. Pretium viverra suspendisse potenti nullam ac tortor vitae purus. Eros donec ac odio tempor orci dapibus ultrices. Elementum nibh tellus molestie nunc. Et magnis dis parturient montes nascetur. Est placerat in egestas erat imperdiet. Consequat interdum varius sit amet mattis vulputate enim.Sit amet nulla facilisi morbi tempus. Nulla facilisi cras fermentum odio eu. Etiam erat velit scelerisque in dictum non consectetur a erat. Enim nulla aliquet porttitor lacus luctus accumsan tortor posuere. Ut sem nulla pharetra diam. Fames ac turpis egestas maecenas. Bibendum neque egestas congue quisque egestas diam. Laoreet id donec ultrices tincidunt arcu non sodales neque. Eget felis eget nunc lobortis mattis aliquam faucibus purus. Faucibus interdum posuere lorem ipsum dolor sit."
    },
    {
      'title': 'STEP 4  드럼 스틱 잡기',
      'subtitle': '드럼 스틱을 잡는 올바른 방법을 익혀보자!',
      'popupTitle': '드럼 스틱 잡는 방법',
      "imagePath": '',
      "description":
          "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Nisl tincidunt eget nullam non. Quis hendrerit dolor magna eget est lorem ipsum dolor sit. Volutpat odio facilisis mauris sit amet massa. Commodo odio aenean sed adipiscing diam donec adipiscing tristique. Mi eget mauris pharetra et. Non tellus orci ac auctor augue. Elit at imperdiet dui accumsan sit. Ornare arcu dui vivamus arcu felis. Egestas integer eget aliquet nibh praesent. In hac habitasse platea dictumst quisque sagittis purus. Pulvinar elementum integer enim neque volutpat ac.Senectus et netus et malesuada. Nunc pulvinar sapien et ligula ullamcorper malesuada proin. Neque convallis a cras semper auctor. Libero id faucibus nisl tincidunt eget. Leo a diam sollicitudin tempor id. A lacus vestibulum sed arcu non odio euismod lacinia. In tellus integer feugiat scelerisque. Feugiat in fermentum posuere urna nec tincidunt praesent. Porttitor rhoncus dolor purus non enim praesent elementum facilisis. Nisi scelerisque eu ultrices vitae auctor eu augue ut lectus. Ipsum faucibus vitae aliquet nec ullamcorper sit amet risus. Et malesuada fames ac turpis egestas sed. Sit amet nisl suscipit adipiscing bibendum est ultricies. Arcu ac tortor dignissim convallis aenean et tortor at. Pretium viverra suspendisse potenti nullam ac tortor vitae purus. Eros donec ac odio tempor orci dapibus ultrices. Elementum nibh tellus molestie nunc. Et magnis dis parturient montes nascetur. Est placerat in egestas erat imperdiet. Consequat interdum varius sit amet mattis vulputate enim.Sit amet nulla facilisi morbi tempus. Nulla facilisi cras fermentum odio eu. Etiam erat velit scelerisque in dictum non consectetur a erat. Enim nulla aliquet porttitor lacus luctus accumsan tortor posuere. Ut sem nulla pharetra diam. Fames ac turpis egestas maecenas. Bibendum neque egestas congue quisque egestas diam. Laoreet id donec ultrices tincidunt arcu non sodales neque. Eget felis eget nunc lobortis mattis aliquam faucibus purus. Faucibus interdum posuere lorem ipsum dolor sit."
    },
  ];

  void _showPopup(String title, String imagePath, String description) {
    showDialog(
      context: context,
      builder: (_) => DrumInfoPopup(
        title: title,
        imagePath: imagePath,
        description: description,
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: EdgeInsets.symmetric(horizontal: 12.w, vertical: 30.h),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // 상단 타이틀
          Center(
            child: Text(
              '드럼 기초',
              style: TextStyle(
                  fontSize: 9.sp,
                  color: Color(0xff595959),
                  fontWeight: FontWeight.w800),
            ),
          ),
          SizedBox(height: 25.h),
          // 카드 리스트
          Expanded(
            child: ListView.builder(
              itemCount: stepInfo.length,
              itemBuilder: (context, index) {
                final step = stepInfo[index];
                final bool isLast = index == stepInfo.length - 1;
                return Padding(
                  padding: EdgeInsets.only(
                    bottom: isLast ? 75.h : 15.h,
                  ),
                  child: Container(
                    decoration: BoxDecoration(
                      color: Colors.white,
                      borderRadius: BorderRadius.circular(12.0),
                      boxShadow: [
                        BoxShadow(
                          color: Colors.grey.withOpacity(0.2),
                          blurRadius: 6,
                          offset: Offset(0, 3),
                        )
                      ],
                    ),
                    child: Row(
                      children: [
                        // 왼쪽 포인트 바
                        Container(
                          width: 4.5.w,
                          height: 93.h,
                          decoration: const BoxDecoration(
                            color: Color(0xFFD97D6C),
                            borderRadius: BorderRadius.only(
                              topLeft: Radius.circular(10),
                              bottomLeft: Radius.circular(10),
                            ),
                          ),
                        ),

                        // 텍스트 내용
                        Expanded(
                          child: Padding(
                            padding: EdgeInsets.only(
                              left: 10.w,
                              top: 17.h,
                              bottom: 18.h,
                            ),
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Text(
                                  step['title']!,
                                  style: TextStyle(
                                    fontWeight: FontWeight.w700,
                                    fontSize: 8.sp,
                                    color: Color(0xFF595959),
                                  ),
                                ),
                                Text(
                                  step['subtitle']!,
                                  style: TextStyle(
                                    fontSize: 6.sp,
                                    color: Color(0xFF949494),
                                  ),
                                ),
                              ],
                            ),
                          ),
                        ),

                        // 오른쪽 > 버튼
                        IconButton(
                          icon: const Icon(Icons.chevron_right),
                          onPressed: () => _showPopup(step['popupTitle']!,
                              step['imagePath']!, step['description']!),
                          color: Colors.black45,
                          iconSize: 15.sp,
                          padding: EdgeInsets.only(right: 10.w),
                        ),
                      ],
                    ),
                  ),
                );
              },
            ),
          ),
        ],
      ),
    );
  }
}
