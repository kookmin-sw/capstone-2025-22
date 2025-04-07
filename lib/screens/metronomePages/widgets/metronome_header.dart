import 'package:flutter/material.dart';

class MetronomeHeader extends StatefulWidget {
  final VoidCallback onBackPressed;
  final VoidCallback onPlayPressed;
  final bool isPlaying;
  final bool isHitVisible;
  final bool isSoundOn;
  final Function(bool) onHitVisibleChanged;
  final Function(bool) onSoundOnChanged;

  const MetronomeHeader({
    required this.onBackPressed,
    required this.onPlayPressed,
    required this.isPlaying,
    required this.isHitVisible,
    required this.isSoundOn,
    required this.onHitVisibleChanged,
    required this.onSoundOnChanged,
    super.key,
  });

  @override
  State<MetronomeHeader> createState() => _MetronomeHeaderState();
}

class _MetronomeHeaderState extends State<MetronomeHeader> {
  final GlobalKey _popupMenuKey = GlobalKey();

  void _updatePopupMenu() {
    (_popupMenuKey.currentState as dynamic).showButtonMenu();
  }

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(top: 25),
      child: Row(
        children: [
          Expanded(
            flex: 1,
            child: IconButton(
              onPressed: widget.onBackPressed,
              icon: const Icon(
                Icons.arrow_back_ios,
                size: 35,
              ),
            ),
          ),
          const Spacer(flex: 4),
          Align(
            alignment: Alignment.center,
            child: Container(
              height: 50,
              width: 150,
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(30),
                boxShadow: [
                  BoxShadow(
                    color: Colors.black.withOpacity(0.1),
                    blurRadius: 10,
                    spreadRadius: 1,
                    offset: const Offset(0, 4),
                  ),
                ],
              ),
              child: Transform.translate(
                offset: const Offset(0, -12),
                child: IconButton(
                  onPressed: widget.onPlayPressed,
                  icon: Icon(
                    widget.isPlaying ? Icons.stop : Icons.play_arrow,
                    size: 60,
                  ),
                  highlightColor: Colors.transparent,
                  color: const Color(0xffD97D6C),
                ),
              ),
            ),
          ),
          const Spacer(flex: 4),
          Expanded(
            flex: 1,
            child: PopupMenuButton<int>(
              icon: const Icon(
                Icons.more_vert,
                size: 35,
              ),
              offset: const Offset(-40, 50),
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(12),
              ),
              itemBuilder: (context) => [
                PopupMenuItem(
                  enabled: false,
                  child: StatefulBuilder(
                    builder: (context, setStateInternal) => Row(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Container(
                          width: 200,
                          child: Row(
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: [
                              const Text(
                                "타격 표시",
                                style: TextStyle(
                                  fontSize: 17,
                                  color: Colors.black38,
                                ),
                              ),
                              const SizedBox(width: 20),
                              Transform.scale(
                                scaleX: 1.3,
                                scaleY: 1.2,
                                child: Switch(
                                  value: widget.isHitVisible,
                                  activeColor: const Color(0xffD97D6C),
                                  inactiveThumbColor: Colors.grey,
                                  inactiveTrackColor: Colors.grey.shade400,
                                  onChanged: (val) {
                                    widget.onHitVisibleChanged(val);
                                    setStateInternal(() {});
                                  },
                                ),
                              ),
                            ],
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
                PopupMenuItem(
                  enabled: false,
                  child: StatefulBuilder(
                    builder: (context, setStateInternal) => Row(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Container(
                          width: 200,
                          child: Padding(
                            padding: const EdgeInsets.only(top: 10),
                            child: Row(
                              mainAxisAlignment: MainAxisAlignment.center,
                              children: [
                                const Text(
                                  "소리 재생",
                                  style: TextStyle(
                                    fontSize: 17,
                                    color: Colors.black38,
                                  ),
                                ),
                                const SizedBox(width: 20),
                                Transform.scale(
                                  scaleX: 1.3,
                                  scaleY: 1.2,
                                  child: Switch(
                                    value: widget.isSoundOn,
                                    activeColor: const Color(0xffD97D6C),
                                    inactiveThumbColor: Colors.grey,
                                    inactiveTrackColor: Colors.grey.shade400,
                                    onChanged: (val) {
                                      widget.onSoundOnChanged(val);
                                      setStateInternal(() {});
                                    },
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
        ],
      ),
    );
  }
}
