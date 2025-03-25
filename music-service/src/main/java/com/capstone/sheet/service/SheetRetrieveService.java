package com.capstone.sheet.service;

import com.capstone.practice.entity.SheetPractice;
import com.capstone.practice.repository.SheetPracticeRepository;
import com.capstone.sheet.dto.SheetResponseDto;
import com.capstone.sheet.repository.SheetRepository;
import com.capstone.sheet.repository.UserSheetRepository;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;
import java.util.stream.Stream;

@Service
public class SheetRetrieveService {
    private final SheetPracticeRepository sheetPracticeRepository;
    private final UserSheetRepository userSheetRepository;
    public SheetRetrieveService(UserSheetRepository userSheetRepository, SheetPracticeRepository sheetPracticeRepository) {
        this.userSheetRepository = userSheetRepository;
        this.sheetPracticeRepository = sheetPracticeRepository;
    }
    /**
     * 사용자의 모든 악보 정보 조회
     * @param email 사용자 이메일
     * @return SheetResponseDto 목록
    * */
    @Transactional(readOnly = true)
    public List<SheetResponseDto> getSheetsByEmail(String email) {
        Stream<SheetResponseDto> userSheets = userSheetRepository.findAllByEmail(email).stream().map(userSheet -> {
            Pageable pageable = PageRequest.of(0, 1);
            List<SheetPractice> lastPractice = sheetPracticeRepository.findAllByEmailAndSheetId(email, userSheet.getSheetId(), pageable);
            return lastPractice.isEmpty() ? SheetResponseDto.from(userSheet) : SheetResponseDto.from(userSheet, lastPractice.get(0).getCreatedDate());
        });
        return userSheets.toList();
    }
}
