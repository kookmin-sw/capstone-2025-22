import 'package:flutter/material.dart';

class MetronomeControls extends StatelessWidget {
  final String selectedTimeSignature;
  final int curBPM;
  final TextEditingController bpmController;
  final FocusNode bpmFocusNode;
  final Function(String) onTimeSignatureChanged;
  final Function(int) onBPMChanged;
  final VoidCallback onBeatChanged;

  const MetronomeControls({
    required this.selectedTimeSignature,
    required this.curBPM,
    required this.bpmController,
    required this.bpmFocusNode,
    required this.onTimeSignatureChanged,
    required this.onBPMChanged,
    required this.onBeatChanged,
    super.key,
  });

  Widget _buildMenuItem(
      BuildContext context, String data, VoidCallback action) {
    return InkWell(
      onTap: () => {Navigator.pop(context), action()},
      child: Padding(
        padding: const EdgeInsets.symmetric(vertical: 12),
        child: Text(
          data,
          style: const TextStyle(fontSize: 18, fontWeight: FontWeight.w500),
        ),
      ),
    );
  }

  void _showCustomModal(BuildContext context) {
    showDialog(
      context: context,
      barrierColor: Colors.black54,
      anchorPoint: const Offset(100, 300),
      builder: (BuildContext context) {
        return Dialog(
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(15),
          ),
          child: Container(
            width: 300,
            padding: const EdgeInsets.symmetric(vertical: 15, horizontal: 20),
            decoration: BoxDecoration(
              color: Colors.white,
              borderRadius: BorderRadius.circular(15),
            ),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceAround,
                  children: [
                    Column(
                      children: [
                        _buildMenuItem(context, "1/4",
                            () => onTimeSignatureChanged("1/4")),
                        _buildMenuItem(context, "2/4",
                            () => onTimeSignatureChanged("2/4")),
                        _buildMenuItem(context, "3/4",
                            () => onTimeSignatureChanged("3/4")),
                        _buildMenuItem(context, "4/4",
                            () => onTimeSignatureChanged("4/4")),
                      ],
                    ),
                    SizedBox(
                      height: 185,
                      child: VerticalDivider(
                        thickness: 4,
                        color: const Color(0xff494949).withOpacity(0.5),
                      ),
                    ),
                    Column(
                      children: [
                        _buildMenuItem(context, "3/8",
                            () => onTimeSignatureChanged("3/8")),
                        _buildMenuItem(context, "6/8",
                            () => onTimeSignatureChanged("6/8")),
                        _buildMenuItem(context, "9/8",
                            () => onTimeSignatureChanged("9/8")),
                        _buildMenuItem(context, "12/8",
                            () => onTimeSignatureChanged("12/8")),
                      ],
                    ),
                  ],
                ),
              ],
            ),
          ),
        );
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        const Spacer(flex: 2),
        Expanded(
          flex: 2,
          child: TextButton(
            onPressed: () => _showCustomModal(context),
            child: Text(
              selectedTimeSignature,
              style: const TextStyle(
                fontSize: 35,
                fontWeight: FontWeight.w600,
                color: Color(0xff424242),
              ),
            ),
          ),
        ),
        const Spacer(flex: 1),
        Expanded(
          flex: 4,
          child: Padding(
            padding: const EdgeInsets.symmetric(vertical: 30),
            child: Container(
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(20),
                boxShadow: [
                  BoxShadow(
                    color: Colors.grey.withOpacity(0.5),
                    blurRadius: 10,
                    spreadRadius: 1,
                    offset: const Offset(3, 3),
                  ),
                ],
              ),
              child: Column(
                children: [
                  const SizedBox(height: 10),
                  const Text(
                    'BPM',
                    style: TextStyle(
                      fontSize: 25,
                      fontWeight: FontWeight.w400,
                    ),
                  ),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                    children: [
                      IconButton(
                        onPressed: curBPM <= 10
                            ? null
                            : () => onBPMChanged(curBPM - 1),
                        icon: const Icon(Icons.remove, size: 50),
                      ),
                      SizedBox(
                        width: 100,
                        height: 50,
                        child: TextField(
                          controller: bpmController,
                          focusNode: bpmFocusNode,
                          textAlign: TextAlign.center,
                          showCursor: false,
                          decoration:
                              const InputDecoration(border: InputBorder.none),
                          style: TextStyle(
                            fontSize: 50,
                            fontWeight: FontWeight.bold,
                            color: bpmFocusNode.hasFocus
                                ? const Color(0xffD97D6C)
                                : const Color(0xff424242),
                          ),
                          keyboardType: TextInputType.number,
                          onSubmitted: (value) {
                            final bpm = int.tryParse(value);
                            if (bpm != null && bpm >= 10 && bpm <= 400) {
                              onBPMChanged(bpm);
                            } else {
                              bpmController.text = curBPM.toString();
                            }
                          },
                        ),
                      ),
                      IconButton(
                        onPressed: curBPM >= 400
                            ? null
                            : () => onBPMChanged(curBPM + 1),
                        icon: const Icon(Icons.add, size: 50),
                      ),
                    ],
                  ),
                ],
              ),
            ),
          ),
        ),
        const Spacer(flex: 1),
        Expanded(
          flex: 2,
          child: IconButton(
            onPressed: onBeatChanged,
            icon: const Icon(Icons.music_note, size: 65),
          ),
        ),
        const Spacer(flex: 2),
      ],
    );
  }
}
