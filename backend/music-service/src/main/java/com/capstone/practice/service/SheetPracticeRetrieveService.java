package com.capstone.practice.service;

import com.capstone.exception.DataNotFoundException;
import com.capstone.practice.dto.SheetPracticeDetailResponseDto;
import com.capstone.practice.dto.SheetPracticeRepresentResponse;
import com.capstone.practice.dto.SheetPracticeResponseDto;
import com.capstone.practice.entity.SheetPractice;
import com.capstone.practice.repository.SheetPracticeRepository;
import com.capstone.sheet.entity.UserSheet;
import com.capstone.sheet.repository.UserSheetRepository;
import org.springframework.data.domain.PageRequest;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class SheetPracticeRetrieveService {
    private final SheetPracticeRepository sheetPracticeRepository;
    private final UserSheetRepository userSheetRepository;
    public SheetPracticeRetrieveService(SheetPracticeRepository sheetPracticeRepository, UserSheetRepository userSheetRepository) {
        this.sheetPracticeRepository = sheetPracticeRepository;
        this.userSheetRepository = userSheetRepository;
    }
    /**
     * 특정 사용자의 특정 악보 연습 기록을 모두 조회 (날짜순 내림차순 정렬)
     * @param userEmail 사용자 이메일
     * @param pageNumber 조회할 기록의 최대 수
     * @param pageSize pageNumber*pageSize 번째 기록부터 조회
     * @param userSheetId 악보 id
     * @return 악보 연습 기록 목록 (최대 수 : pageNumber)
    * */
    public List<SheetPracticeResponseDto> getSheetPracticeRecords(
            String userEmail,
            int pageNumber,
            int pageSize,
            int userSheetId){
        return sheetPracticeRepository
                .findAllByEmailAndSheetId(userEmail, userSheetId, PageRequest.of(pageNumber, pageSize))
                .stream().map(SheetPracticeResponseDto::from).toList();
    }
    /**
     * 특정 연습 기록의 상세 정보 조회
     * @param sheetPracticeId 악보 연습 id
     * @return 악보 연습 상세 정보
     * @throws DataNotFoundException 해당 악보 연습 id에 해당하는 데이터가 없으면 예외 던짐
    * */
    public SheetPracticeDetailResponseDto getDetailSheetPracticeRecord(int sheetPracticeId){
        SheetPractice sheetPractice = sheetPracticeRepository.findById(sheetPracticeId)
                .orElseThrow(() -> new DataNotFoundException("SheetPractice not found"));
        return SheetPracticeDetailResponseDto.from(sheetPractice);
    }
    /**
     * 특정 악보 연습에 대한 대표 연습 정보 조회
    * */
    public SheetPracticeRepresentResponse getRepresentSheetPractice(String userEmail, int userSheetId){
        UserSheet userSheet = userSheetRepository.findById(userSheetId).orElseThrow(() -> new DataNotFoundException("user sheet not found"));
        return sheetPracticeRepository.findLastPracticeByEmailAndSheetId(userSheetId).map(sheetPractice -> {
            Integer maxScore = sheetPracticeRepository.findMaxScoreById(userSheet.getUserSheetId())
                    .orElseThrow(() -> new DataNotFoundException("exception on getRepresentSheetPractice : sheet practice exists but cant find max score"));
            return SheetPracticeRepresentResponse.from(userSheet, maxScore, sheetPractice.getCreatedDate());
        }).orElse(null);
    }
    /** 
     * 대표 악보 연습 정보 목록 조회
     * @param userEmail 사용자 이메일
     * @return List of SheetPracticeRepresentResponse
    * */
    public List<SheetPracticeRepresentResponse> getRepresentSheetPractices(String userEmail){
        return userSheetRepository.findAllByEmail(userEmail)
                .stream().map(userSheet -> getRepresentSheetPractice(userEmail, userSheet.getUserSheetId())).toList();
    }
}
