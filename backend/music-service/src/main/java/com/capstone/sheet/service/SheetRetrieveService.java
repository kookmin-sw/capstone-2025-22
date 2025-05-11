package com.capstone.sheet.service;

import com.capstone.exception.DataNotFoundException;
import com.capstone.practice.entity.SheetPractice;
import com.capstone.practice.repository.SheetPracticeRepository;
import com.capstone.sheet.dto.SheetDetailResponseDto;
import com.capstone.sheet.dto.SheetResponseDto;
import com.capstone.dto.musicXml.MeasureInfo;
import com.capstone.dto.musicXml.PartInfo;
import com.capstone.sheet.entity.UserSheet;
import com.capstone.sheet.repository.UserSheetRepository;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.ws.rs.InternalServerErrorException;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;
import java.util.stream.Stream;

@Slf4j
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
    /**
     * 특정 악보의 특정 마디 정보 조회
    * */
    public MeasureInfo findMeasureInfo(int userSheetId, String measureNumber){
        try {
            UserSheet userSheet = userSheetRepository.findById(userSheetId)
                    .orElseThrow(() -> new DataNotFoundException("UserSheet Not Found"));
            String sheetJson = userSheet.getSheet().getSheetJson();
            List<PartInfo> partInfoList = new ObjectMapper().readValue(sheetJson, new TypeReference<>() {});
            List<MeasureInfo> measureList = partInfoList.get(0).getMeasureList();
            for(MeasureInfo measure : measureList){
                if(measure.getMeasureNumber().equals(measureNumber)){
                    return measure;
                }
            }
            throw new DataNotFoundException("Measure Not Found");
        }catch (JsonProcessingException e){
            log.error(e.getMessage(), e);
            throw new InternalServerErrorException("Json Mapping Error");
        }
    }
}
