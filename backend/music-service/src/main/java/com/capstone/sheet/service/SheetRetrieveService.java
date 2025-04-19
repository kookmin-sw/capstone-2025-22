package com.capstone.sheet.service;

import com.capstone.exception.DataNotFoundException;
import com.capstone.practice.entity.SheetPractice;
import com.capstone.practice.repository.SheetPracticeRepository;
import com.capstone.sheet.dto.SheetDetailResponseDto;
import com.capstone.sheet.dto.SheetResponseDto;
import com.capstone.sheet.entity.Sheet;
import com.capstone.sheet.entity.UserSheet;
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
            List<SheetPractice> lastPractice = sheetPracticeRepository.findAllByEmailAndSheetId(email, userSheet.getUserSheetId(), pageable);
            return lastPractice.isEmpty() ? SheetResponseDto.from(userSheet) : SheetResponseDto.from(userSheet, lastPractice.get(0).getCreatedDate());
        });
        return userSheets.toList();
    }
    /**
     * 특정 악보 상세 정보 조회
     * @param userSheetId 악보 id
     * @return SheetDetailResponseDto
     * @throws DataNotFoundException if UserSheet not exists
    * */
    public SheetDetailResponseDto getSheetById(int userSheetId) {
        UserSheet userSheet = userSheetRepository.findById(userSheetId)
                .orElseThrow(() -> new DataNotFoundException("UserSheet not found"));
        SheetPractice lastPractice = sheetPracticeRepository.findLastPracticeByEmailAndSheetId(userSheetId)
                .orElse(null);
        return lastPractice==null ?
                SheetDetailResponseDto.from(userSheet, null) :
                SheetDetailResponseDto.from(userSheet, lastPractice.getCreatedDate());
    }
}
