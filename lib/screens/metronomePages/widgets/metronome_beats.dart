import 'package:flutter/material.dart';

class MetronomeBeats extends StatefulWidget {
  final int currentBeat;
  final String timeSignature;
  final bool currItemBase4;
  final bool isSoundOn;
  final bool isPlaying;

  const MetronomeBeats({
    required this.currentBeat,
    required this.timeSignature,
    required this.currItemBase4,
    required this.isSoundOn,
    required this.isPlaying,
    super.key,
  });

  @override
  State<MetronomeBeats> createState() => _MetronomeBeatsState();
}

class _MetronomeBeatsState extends State<MetronomeBeats>
    with SingleTickerProviderStateMixin {
  late AnimationController _animationController;
  late Animation<double> _animation;

  @override
  void initState() {
    super.initState();
    _animationController = AnimationController(
      duration: const Duration(milliseconds: 100),
      vsync: this,
    );
    _animation = Tween<double>(begin: 0.0, end: 1.0).animate(
      CurvedAnimation(
        parent: _animationController,
        curve: Curves.easeOut,
      ),
    );
  }

  @override
  void didUpdateWidget(MetronomeBeats oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (widget.isPlaying &&
        widget.currItemBase4 &&
        widget.timeSignature == '1/4') {
      _animationController.forward(from: 0.0);
    }
  }

  @override
  void dispose() {
    _animationController.dispose();
    super.dispose();
  }

  Widget metronomeItem(int idx) {
    bool isSelected = (idx == widget.currentBeat);
    bool shouldShowBloom = isSelected && widget.isPlaying;

    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 50),
      child: Stack(
        clipBehavior: Clip.none,
        children: [
          if (shouldShowBloom)
            (widget.currItemBase4 && widget.timeSignature == '1/4')
                ? AnimatedBuilder(
                    animation: _animation,
                    builder: (context, child) {
                      return Positioned(
                        left: -5,
                        top: -5,
                        child: Opacity(
                          opacity: _animation.value,
                          child: Container(
                            width: 80,
                            height: 80,
                            decoration: BoxDecoration(
                              color: const Color(0xffB95D4C),
                              borderRadius: BorderRadius.circular(100),
                              boxShadow: [
                                BoxShadow(
                                  color:
                                      const Color(0xffD97D6C).withOpacity(0.7),
                                  blurRadius: 5,
                                  spreadRadius: 5,
                                ),
                              ],
                            ),
                          ),
                        ),
                      );
                    },
                  )
                : Positioned(
                    left: -5,
                    top: -5,
                    child: Container(
                      width: (!widget.currItemBase4 && idx % 3 != 0) ? 60 : 80,
                      height: (!widget.currItemBase4 && idx % 3 != 0) ? 60 : 80,
                      decoration: BoxDecoration(
                        color: const Color(0xffB95D4C),
                        borderRadius: BorderRadius.circular(100),
                        boxShadow: [
                          BoxShadow(
                            color: const Color(0xffD97D6C).withOpacity(0.7),
                            blurRadius: 5,
                            spreadRadius: 5,
                          ),
                        ],
                      ),
                    ),
                  ),
          Container(
            width: (!widget.currItemBase4 && idx % 3 != 0) ? 50 : 70,
            height: (!widget.currItemBase4 && idx % 3 != 0) ? 50 : 70,
            decoration: BoxDecoration(
              color: isSelected
                  ? const Color(0xffF28C79)
                  : (!widget.currItemBase4 && idx % 3 != 0)
                      ? const Color(0xff8C7571)
                      : const Color.fromARGB(255, 123, 87, 81),
              borderRadius: BorderRadius.circular(100),
            ),
          ),
        ],
      ),
    );
  }

  List<Widget> generateMetronomeItems(int count) {
    List<Widget> items = List.generate(count, (index) => metronomeItem(index));

    if (widget.timeSignature != '9/8' && widget.timeSignature != '12/8') {
      return [
        Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: items,
        )
      ];
    } else {
      if (widget.timeSignature == '9/8') {
        return [
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: items.sublist(0, 3),
          ),
          const SizedBox(height: 20),
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: items.sublist(3, 9),
          ),
        ];
      } else {
        return [
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: items.sublist(0, 6),
          ),
          const SizedBox(height: 20),
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: items.sublist(6, 12),
          ),
        ];
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    int count = int.parse(widget.timeSignature.split('/')[0]);
    return Container(
      decoration: const BoxDecoration(
        color: Color(0xff424242),
      ),
      child: Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: generateMetronomeItems(count),
        ),
      ),
    );
  }
}
