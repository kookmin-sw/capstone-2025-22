import 'package:flutter/material.dart';
import 'package:file_selector/file_selector.dart';
import 'dart:math';
import 'dart:io';
import 'dart:async';
import 'package:flutter_screenutil/flutter_screenutil.dart';

class AddSheetDialog extends StatefulWidget {
  final Function(String, String, String?) onSubmit;

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
          width: 120.w,
          padding: EdgeInsets.symmetric(horizontal: 5.w, vertical: 20.h),
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
        Text(
          '악보 파일 업로드',
          style: TextStyle(
            fontSize: 7.sp,
            fontWeight: FontWeight.w600,
            color: Colors.black87,
          ),
          textAlign: TextAlign.center,
        ),
        SizedBox(height: 5.h),
        Text(
          '연주할 악보 PDF 파일을 선택해주세요.',
          style: TextStyle(
            fontSize: 5.sp,
            color: Colors.black54,
          ),
          textAlign: TextAlign.center,
        ),
        SizedBox(height: 20.h),
        _selectedFilePath == null
            ? Center(
                child: ElevatedButton(
                  style: ElevatedButton.styleFrom(
                    backgroundColor: const Color(0xFFD97D6C),
                    minimumSize: Size(60.w, 50.h),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12),
                    ),
                  ),
                  onPressed: _uploadPDF,
                  child: Text(
                    '파일 선택',
                    style: TextStyle(
                      color: Colors.white,
                      fontWeight: FontWeight.w600,
                      fontSize: 5.5.sp,
                    ),
                  ),
                ),
              )
            : Container(
                margin: EdgeInsets.symmetric(horizontal: 5.w),
                padding: EdgeInsets.symmetric(horizontal: 5.w, vertical: 10.h),
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(8),
                  border: Border.all(
                      color: const Color.fromARGB(255, 149, 148, 148)),
                ),
                child: Row(
                  children: [
                    Icon(
                      Icons.insert_drive_file_outlined,
                      color: Color(0xFFE57373),
                      size: 10.sp,
                    ),
                    SizedBox(width: 7.w),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          Text(
                            '${_sheetNameController.text}.pdf',
                            style: TextStyle(
                              fontSize: 5.5.sp,
                              color: Colors.black87,
                            ),
                          ),
                          SizedBox(height: 5.h),
                          Text(
                            _fileSize,
                            style: TextStyle(
                              fontSize: 4.8.sp,
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
                      child: Icon(
                        Icons.delete,
                        color: Colors.black38,
                        size: 9.sp,
                      ),
                    ),
                  ],
                ),
              ),
        SizedBox(height: 25.h),
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
        Text(
          '악보명을 입력해주세요.',
          style: TextStyle(
            fontSize: 6.sp,
            fontWeight: FontWeight.w600,
            color: Color(0xFF646464),
          ),
          textAlign: TextAlign.center,
        ),
        SizedBox(height: 24.h),
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
            decoration: InputDecoration(
              hintText: '악보명',
              hintStyle: TextStyle(
                color: Colors.black45,
                fontSize: 5.sp,
              ),
              contentPadding:
                  EdgeInsets.symmetric(horizontal: 8.w, vertical: 12.h),
              border: InputBorder.none,
            ),
          ),
        ),
        SizedBox(height: 25.h),
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
        Text(
          '가수명을 입력해주세요.',
          style: TextStyle(
            fontSize: 6.sp,
            fontWeight: FontWeight.w600,
            color: Color(0xFF646464),
          ),
          textAlign: TextAlign.center,
        ),
        SizedBox(height: 24.h),
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
            decoration: InputDecoration(
              hintText: '가수명',
              hintStyle: TextStyle(
                color: Colors.black45,
                fontSize: 5.sp,
              ),
              contentPadding:
                  EdgeInsets.symmetric(horizontal: 8.w, vertical: 12.h),
              border: InputBorder.none,
            ),
          ),
        ),
        SizedBox(height: 25.h),
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
                      _selectedFilePath // 파일 경로
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
      padding: EdgeInsets.symmetric(horizontal: 12.h),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          SizedBox(
            width: 48.w,
            child: TextButton(
              onPressed: onPrevious,
              style: TextButton.styleFrom(
                backgroundColor: const Color(0xFFF5F5F5),
                padding: EdgeInsets.symmetric(vertical: 17.h),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(8),
                ),
              ),
              child: Text(
                previousText,
                style: TextStyle(
                  color: Colors.black54,
                  fontSize: 6.sp,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ),
          ),
          SizedBox(
            width: 48.w,
            child: TextButton(
              onPressed: onNext,
              style: TextButton.styleFrom(
                backgroundColor: const Color(0xFFD97D6C),
                padding: EdgeInsets.symmetric(vertical: 17.h),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(8),
                ),
                disabledBackgroundColor:
                    const Color.fromARGB(255, 136, 135, 135).withOpacity(0.5),
              ),
              child: Text(
                nextText,
                style: TextStyle(
                  color: Colors.white,
                  fontSize: 6.sp,
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
