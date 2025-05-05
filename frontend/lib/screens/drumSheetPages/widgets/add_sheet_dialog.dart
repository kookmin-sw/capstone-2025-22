import 'package:flutter/material.dart';
import 'package:file_selector/file_selector.dart';
import 'dart:math';
import 'dart:io';
import 'dart:async';

class AddSheetDialog extends StatefulWidget {
  final Function(String, String) onSubmit;

  const AddSheetDialog({super.key, required this.onSubmit});

  @override
  State<AddSheetDialog> createState() => _AddSheetDialogState();
}

class _AddSheetDialogState extends State<AddSheetDialog> {
  final TextEditingController _sheetNameController = TextEditingController();
  final TextEditingController _artistNameController = TextEditingController();

  bool _isNextButtonEnabled = false; // 완료 버튼 상태 관리

  String? _selectedFilePath;
  String _fileSize = '';
  int _currentStep = 0;

  @override
  void dispose() {
    _sheetNameController.dispose();
    _artistNameController.dispose();
    super.dispose();
  }

  // 값이 변경될 때마다 실시간으로 상태를 업데이트
  void _onTextChanged() {
    setState(() {
      // 악보명과 가수명이 모두 입력되었을 때만 '완료' 버튼을 활성화
      _isNextButtonEnabled = _sheetNameController.text.isNotEmpty &&
          _artistNameController.text.isNotEmpty;
    });
  }

  String _formatFileSize(int bytes) {
    if (bytes <= 0) return '0B';
    const suffixes = ['B', 'KB', 'MB', 'GB'];
    var i = (log(bytes) / log(1024)).floor();
    return '${(bytes / pow(1024, i)).toStringAsFixed(2)}${suffixes[i]}';
  }

  Future<void> _uploadPDF() async {
    try {
      final XTypeGroup pdfGroup = XTypeGroup(
        label: 'PDFs',
        extensions: ['pdf'],
      );

      final XFile? file = await openFile(
        acceptedTypeGroups: [pdfGroup],
      );

      if (file != null && mounted) {
        final fileSize = await File(file.path).length();
        setState(() {
          _selectedFilePath = file.path;
          _sheetNameController.text = file.name.replaceAll('.pdf', '');
          _fileSize = _formatFileSize(fileSize);
        });
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('파일 선택 중 오류가 발생했습니다: ${e.toString()}')),
        );
      }
    }
  }

  void _clearSelectedFile() {
    setState(() {
      _selectedFilePath = null;
      _sheetNameController.text = '';
      _fileSize = '';
    });
  }

  @override
  Widget build(BuildContext context) {
    return Dialog(
      backgroundColor: Colors.white,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(16),
      ),
      child: SingleChildScrollView(
        child: Container(
          width: 330,
          padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 20),
          decoration: BoxDecoration(
            color: Colors.white,
            borderRadius: BorderRadius.circular(16),
          ),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              if (_currentStep == 0)
                _buildFileSelectionStep()
              else if (_currentStep == 1)
                _buildSheetNameStep()
              else
                _buildArtistNameStep(),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildFileSelectionStep() {
    return Column(
      mainAxisSize: MainAxisSize.min,
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        const Text(
          '악보 파일 업로드',
          style: TextStyle(
            fontSize: 18,
            fontWeight: FontWeight.w600,
            color: Colors.black87,
          ),
          textAlign: TextAlign.center,
        ),
        const SizedBox(height: 5),
        const Text(
          '연주할 악보 PDF 파일을 선택해주세요.',
          style: TextStyle(
            fontSize: 14,
            color: Colors.black54,
          ),
          textAlign: TextAlign.center,
        ),
        const SizedBox(height: 20),
        _selectedFilePath == null
            ? Center(
                child: ElevatedButton(
                  style: ElevatedButton.styleFrom(
                    backgroundColor: const Color(0xFFD97D6C),
                    minimumSize: const Size(140, 35),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12),
                    ),
                  ),
                  onPressed: _uploadPDF,
                  child: const Text(
                    '파일 선택',
                    style: TextStyle(
                      color: Colors.white,
                      fontWeight: FontWeight.w600,
                      fontSize: 14,
                    ),
                  ),
                ),
              )
            : Container(
                margin: const EdgeInsets.symmetric(horizontal: 20),
                padding:
                    const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(8),
                  border: Border.all(
                      color: const Color.fromARGB(255, 149, 148, 148)),
                ),
                child: Row(
                  children: [
                    const Icon(
                      Icons.insert_drive_file_outlined,
                      color: Color(0xFFE57373),
                      size: 35,
                    ),
                    const SizedBox(width: 12),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          Text(
                            '${_sheetNameController.text}.pdf',
                            style: const TextStyle(
                              fontSize: 16,
                              color: Colors.black87,
                            ),
                          ),
                          Text(
                            _fileSize,
                            style: const TextStyle(
                              fontSize: 12,
                              color: Colors.black38,
                              letterSpacing: -0.5,
                              height: 1,
                            ),
                          ),
                        ],
                      ),
                    ),
                    GestureDetector(
                      onTap: _clearSelectedFile,
                      child: const Icon(
                        Icons.delete,
                        color: Colors.black38,
                        size: 20,
                      ),
                    ),
                  ],
                ),
              ),
        const SizedBox(height: 20),
        _buildNavigationButtons(
          onPrevious: () => Navigator.of(context).pop(),
          previousText: '취소',
          onNext: _selectedFilePath != null
              ? () {
                  setState(() {
                    _currentStep++;
                  });
                }
              : null,
          nextText: '다음',
        ),
      ],
    );
  }

  Widget _buildSheetNameStep() {
    return Column(
      mainAxisSize: MainAxisSize.min,
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        const Text(
          '악보명을 입력해주세요.',
          style: TextStyle(
            fontSize: 16,
            fontWeight: FontWeight.w600,
            color: Color(0xFF646464),
          ),
          textAlign: TextAlign.center,
        ),
        const SizedBox(height: 24),
        Container(
          decoration: BoxDecoration(
            border: Border.all(color: const Color(0xFFD5D5D5)),
            borderRadius: BorderRadius.circular(8),
          ),
          child: TextField(
            controller: _sheetNameController,
            keyboardType: TextInputType.text,
            textInputAction: TextInputAction.next,
            onChanged: (text) {
              setState(() {}); // 값 변경 시 상태를 즉시 갱신
            },
            decoration: const InputDecoration(
              hintText: '악보명',
              hintStyle: TextStyle(
                color: Colors.black45,
                fontSize: 14,
              ),
              contentPadding:
                  EdgeInsets.symmetric(horizontal: 16, vertical: 12),
              border: InputBorder.none,
            ),
          ),
        ),
        const SizedBox(height: 20),
        _buildNavigationButtons(
          onPrevious: () {
            setState(() {
              _currentStep--;
            });
          },
          previousText: '이전',
          onNext: _sheetNameController.text.isNotEmpty
              ? () {
                  setState(() {
                    _currentStep++;
                  });
                }
              : null,
          nextText: '다음',
        ),
      ],
    );
  }

  Widget _buildArtistNameStep() {
    return Column(
      mainAxisSize: MainAxisSize.min,
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        const Text(
          '가수명을 입력해주세요.',
          style: TextStyle(
            fontSize: 16,
            fontWeight: FontWeight.w600,
            color: Color(0xFF646464),
          ),
          textAlign: TextAlign.center,
        ),
        const SizedBox(height: 24),
        Container(
          decoration: BoxDecoration(
            border: Border.all(color: const Color(0xFFD5D5D5)),
            borderRadius: BorderRadius.circular(8),
          ),
          child: TextField(
            controller: _artistNameController,
            keyboardType: TextInputType.text,
            textInputAction: TextInputAction.done,
            onChanged: (text) {
              setState(() {}); // 값 변경 시 상태를 즉시 갱신
            },
            decoration: const InputDecoration(
              hintText: '가수명',
              hintStyle: TextStyle(
                color: Colors.black45,
                fontSize: 14,
              ),
              contentPadding:
                  EdgeInsets.symmetric(horizontal: 16, vertical: 12),
              border: InputBorder.none,
            ),
          ),
        ),
        const SizedBox(height: 20),
        _buildNavigationButtons(
          onPrevious: () {
            setState(() {
              _currentStep--;
            });
          },
          previousText: '이전',
          onNext: _artistNameController.text.isNotEmpty
              ? () {
                  widget.onSubmit(
                    _sheetNameController.text, // 악보명
                    _artistNameController.text, // 가수명
                  );
                  Navigator.of(context).pop();
                }
              : null,
          nextText: '완료',
        ),
      ],
    );
  }

  Widget _buildNavigationButtons({
    required VoidCallback onPrevious,
    required String previousText,
    required VoidCallback? onNext,
    required String nextText,
  }) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 20),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          SizedBox(
            width: 130,
            child: TextButton(
              onPressed: onPrevious,
              style: TextButton.styleFrom(
                backgroundColor: const Color(0xFFF5F5F5),
                padding: const EdgeInsets.symmetric(vertical: 12),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(8),
                ),
              ),
              child: Text(
                previousText,
                style: const TextStyle(
                  color: Colors.black54,
                  fontSize: 16,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ),
          ),
          SizedBox(
            width: 130,
            child: TextButton(
              onPressed: onNext,
              style: TextButton.styleFrom(
                backgroundColor: const Color(0xFFD97D6C),
                padding: const EdgeInsets.symmetric(vertical: 12),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(8),
                ),
                disabledBackgroundColor:
                    const Color.fromARGB(255, 136, 135, 135).withOpacity(0.5),
              ),
              child: Text(
                nextText,
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 16,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}
