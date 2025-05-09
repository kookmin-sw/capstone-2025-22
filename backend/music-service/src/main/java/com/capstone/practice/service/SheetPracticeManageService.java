package com.capstone.practice.service;

import com.capstone.exception.DataNotFoundException;
import com.capstone.practice.dto.SheetPracticeResponseDto;
import com.capstone.practice.entity.SheetPractice;
import com.capstone.practice.repository.SheetPracticeRepository;
import com.capstone.sheet.entity.UserSheet;
import com.capstone.sheet.repository.UserSheetRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import static com.capstone.dto.sheet.MusicServiceClientDto.*;

@Service
@RequiredArgsConstructor
public class SheetPracticeManageService {
    private final SheetPracticeRepository sheetPracticeRepository;
    private final UserSheetRepository userSheetRepository;

    public SheetPracticeResponseDto saveSheetPractice(SheetPracticeCreateRequest sheetPracticeCreateRequest) {
        try {
            UserSheet userSheet = userSheetRepository.findById(sheetPracticeCreateRequest.getUserSheetId())
                    .orElseThrow(() -> new DataNotFoundException("UserSheet not found"));
            SheetPractice sheetPractice = SheetPractice.from(sheetPracticeCreateRequest, userSheet);
            return SheetPracticeResponseDto.from(sheetPracticeRepository.save(sheetPractice));
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
