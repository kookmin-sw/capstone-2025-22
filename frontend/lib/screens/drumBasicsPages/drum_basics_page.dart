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
      "description": """# 드럼 용어 알기  
드럼 연주를 더 풍부하고 정확하게 표현하기 위해 꼭 알아두어야 할 기본 용어 목록입니다.  


---


## 1. 템포(Tempo)  
템포는 곡의 빠르기를 나타내는 지표이며, BPM(Beats Per Minute) 단위입니다.  

## 2. 박자(Meter / Time Signature)  
박자는 한 마디 안에서 박자를 어떻게 나눌지 결정하는 기호이며, 예를 들어 4/4 박자는 한 마디에 4개의 4분 음표가 들어가는 형식입니다.  

## 3. 다운비트(Downbeat) & 업비트(Upbeat)  
- **다운비트**는 마디의 첫 박자이며 리듬의 시작점입니다.  
- **업비트**는 다음 마디 첫 박자 전의 빠른 박자이며 경쾌한 연결 고리입니다.  

## 4. 백비트(Backbeat)  
백비트는 4/4 박자에서 2번과 4번 박자에 악센트를 주는 패턴입니다.  

## 5. 그루브(Groove)  
그루브는 리듬의 흐름 또는 타이트한 느낌을 의미하는 용어입니다.  

## 6. 필인(Fill-in)  
필인은 마디와 마디 사이를 이어 주는 장식 패턴입니다.  

## 7. 다이내믹(Dynamics / 셈여림)  
다이내믹은 연주 강약을 조절하는 표시입니다.  
- **pp (피아니시모)**: 매우 약한 강도입니다.  
- **mf (메조포르테)**: 중간 정도 강도입니다.  
- **ff (포르티시모)**: 매우 강한 강도입니다.  """
    },
    {
      'title': 'STEP 3  드럼 악보 알기',
      'subtitle': '드럼 악보를 어떻게 읽는지 알아보자!',
      'popupTitle': '드럼 악보 읽기',
      "imagePath": 'assets/images/drum_sheet.png',
      "description": """# 드럼 악보 읽는 방법

드럼 악보는 일반적인 악보와 달리 음의 높이보다 **어떤 악기를 언제 치는지**를 나타냅니다.  
아래 내용을 통해 드럼 악보의 기본 구조와 해석 방법을 알아보세요.


---


## 1. 드럼 악보의 기본 구성

### 1.1 5선 악보 (Staff)
- 다섯 줄로 구성된 기본 악보입니다.
- 각 줄과 줄 사이(선, 선간)는 드럼 세트의 서로 다른 악기를 나타냅니다.

### 1.2 드럼 표기법 (Notation)
- 음표는 점(`●`), 십자(`x`), 원 등으로 표시되며, 위치에 따라 연주할 악기를 구분합니다.

### 1.3 박자 기호 (Time Signature)
- `4/4`, `3/4` 등으로 표기되며, 한 마디에 들어가는 박자의 수와 단위를 나타냅니다.

### 1.4 음표 값 (Note Values)
- 음의 길이를 나타내며, 전음표 → 2분음표 → 4분음표 → 8분음표 → 16분음표 순으로 짧아집니다.

### 1.5 쉼표 (Rests)
- 소리를 내지 않고 쉬는 구간을 표시합니다. 음표와 마찬가지로 길이에 따라 여러 종류가 있습니다.

### 1.6 점음표 (Dotted Notes)
- 음표 오른쪽에 점을 찍어 **원래 길이 + 절반만큼 더 연주**합니다.

### 1.7 잇단음표 (Ties)
- 두 개의 같은 음을 곡선으로 연결해 하나의 음처럼 이어서 연주합니다.

### 1.8 반복 기호 (Repeats)
- `:‖:` 기호로 구간 반복을 지시합니다.


---


## 2. 오선 위의 드럼 악기 위치

| 악기        | 기호 | 위치 설명                     |
|-------------|------|------------------------------|
| 크래시 심벌 | `x`  | 맨 윗선 위 또는 맨 윗선       |
| 라이드 심벌 | `x`  | 맨 윗선 또는 선 위            |
| 하이햇      | `x`  | 윗선 (닫힘/열림 기호로 구분)  |
| 스네어      | `●`  | 가운데 선 (3선)              |
| 하이탐      | `●`  | 3선 위                        |
| 로우탐      | `●`  | 3선 아래                      |
| 플로어탐    | `●`  | 아랫선                        |
| 베이스드럼  | `●`  | 오선 아래 (선 밑 공간 또는 아래 선) |

- 기호 `x` : 심벌 계열 (하이햇, 크래시 등)  
- 기호 `●` : 드럼 계열 (스네어, 킥 등)


---


## 3. 하이햇 연주 기호

| 기호 | 의미                          |
|------|-------------------------------|
| `x`  | 일반 하이햇 연주               |
| `+`  | 클로즈 하이햇 (닫은 상태)      |
| `o`  | 오픈 하이햇 (열린 상태)        |
| `foot` | 발로 연주하는 하이햇 (오선 아래) |


---


## 4. 리듬 악보 예시 (기본 4/4 리듬)

x - x - x - x - ← 하이햇 (오른손)
● ● ← 스네어 (왼손, 2박/4박)
● ● ← 킥 (오른발, 1박/3박)


- `x` : 하이햇
- `●` : 스네어나 킥 (위치로 구분)
- `-` : 쉼표 (연주 없음)


---


## 5. 자주 사용되는 특수 기호

| 기호   | 의미                               |
|--------|------------------------------------|
| `>`    | 악센트 (해당 음을 더 강하게)        |
| `()`   | 유령음 (ghost note, 매우 약하게 연주) |
| `:‖:`  | 반복 연주 지시                        |
| `.`    | 점음표 (길이를 1.5배로 늘림)         |
| `~` 또는 타이곡선 | 잇단음 (두 음을 연결해 하나로 연주) |


---


## TIP

- **같은 기호라도 위치에 따라 다른 악기**를 의미합니다. 기호와 위치를 함께 익혀야 합니다.
- 처음에는 시범 연주나 리듬 머신과 함께 **소리를 들으며 악보를 따라 읽는 연습**이 효과적입니다."""
    },
    {
      'title': 'STEP 4  드럼 스틱 잡기',
      'subtitle': '드럼 스틱을 잡는 올바른 방법을 익혀보자!',
      'popupTitle': '드럼 스틱 잡는 방법',
      "imagePath": 'assets/images/drum_stick.png',
      "description": """## 1. 손의 위치

- 스틱 끝에서 약 1/3 지점을 엄지와 검지로 가볍게 집습니다.  
- 엄지와 검지가 V자 형태를 만들도록 잡습니다.  
- 손에 힘을 주지 않고, 연필을 쥐듯 자연스럽게 잡는 것이 중요합니다.

## 2. 손가락 배치

- 엄지와 검지는 스틱을 고정하는 역할을 합니다.  
- 중지, 약지, 새끼손가락은 스틱을 부드럽게 감싸 받쳐줍니다.  
- 손가락을 스틱에 너무 꽉 붙이지 않고, 탄력 있게 움직일 수 있도록 공간을 둡니다.

## 3. 손목과 팔의 각도

- 팔꿈치는 몸에서 살짝 떨어진 위치에 둡니다.  
- 손목이 편하게 위아래로 움직일 수 있도록 힘을 뺍니다.  
- 손등이 약간 위를 향하게 하여 손목의 자연스러운 움직임을 유도합니다.

## 4. 양손의 균형

- 양손이 대칭되도록 스틱의 잡는 위치, 각도, 손목 자세를 동일하게 유지합니다.  
- 균형 잡힌 자세는 일정한 리듬과 안정적인 연주에 도움이 됩니다.

## TIP

- 거울을 보면서 자세를 확인하면 도움이 됩니다.  
- 손과 팔에 힘이 들어가지 않도록 주의하며 연습합니다.
"""
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
