package com.capstone.sheet.service;

import com.capstone.exception.DataNotFoundException;
import com.capstone.exception.InvalidRequestException;
import com.capstone.practice.entity.SheetPractice;
import com.capstone.practice.repository.SheetPracticeRepository;
import com.capstone.sheet.dto.SheetResponseDto;
import com.capstone.sheet.entity.UserSheet;
import com.capstone.sheet.repository.UserSheetRepository;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;

@Service
public class SheetManageService {
    UserSheetRepository userSheetRepository;
    SheetPracticeRepository sheetPracticeRepository;
    public SheetManageService(
            UserSheetRepository userSheetRepository,
            SheetPracticeRepository sheetPracticeRepository) {
        this.userSheetRepository = userSheetRepository;
        this.sheetPracticeRepository = sheetPracticeRepository;
    }
    /**
     * 사용자 이메일, 악보 id 기반으로 악보 이름 수정
     * @param userEmail 사용자 이메일
     * @param newName 새로운 악보 이름
     * @param userSheetId 악보 id
     * @return SheetResponseDto
     * @throws DataNotFoundException 해당 사용자의 악보가 없는 경우 예외를 던짐
     * @throws InvalidRequestException 악보 주인과 요청 이메일이 불일치하면 예외를 던짐
    * */
    @Transactional
    public SheetResponseDto updateSheetName(String userEmail, String newName, int userSheetId){
        UserSheet userSheet = userSheetRepository.findById(userSheetId)
                .orElseThrow(() -> new DataNotFoundException("User Sheet Not Found"));
        if(!userSheet.getUserEmail().equals(userEmail)){
            throw new InvalidRequestException("Access Denied. You are not allowed to delete this sheet");
        }
        // set new name
        userSheet.setSheetName(newName);
        // get last practice date
        Pageable pageable = PageRequest.of(0, 1);
        List<SheetPractice> lastPractice = sheetPracticeRepository.findAllByEmailAndSheetId(userEmail, userSheetId, pageable);
        return lastPractice.isEmpty() ? SheetResponseDto.from(userSheet) : SheetResponseDto.from(userSheet, lastPractice.get(0).getCreatedDate());
    }
    /**
     * 사용자 이메일, 악보 id 기반으로 악보 색상 수정
     * @param userEmail 사용자 이메일
     * @param color 새로운 악보 색상
     * @param userSheetId 악보 id
     * @return SheetResponseDto
     * @throws DataNotFoundException 해당 사용자의 악보가 없는 경우 예외를 던짐
     * @throws InvalidRequestException 악보 주인과 요청 이메일이 불일치하면 예외를 던짐
     * */
    @Transactional
    public SheetResponseDto updateSheetColor(String userEmail, String color, int userSheetId){
        UserSheet userSheet = userSheetRepository.findById(userSheetId)
                .orElseThrow(() -> new DataNotFoundException("User Sheet Not Found"));
        if(!userSheet.getUserEmail().equals(userEmail)){
            throw new InvalidRequestException("Access Denied. You are not allowed to delete this sheet");
        }
        // set new color
        userSheet.setColor(color);
        // get last practice date
        Pageable pageable = PageRequest.of(0, 1);
        List<SheetPractice> lastPractice = sheetPracticeRepository.findAllByEmailAndSheetId(userEmail, userSheetId, pageable);
        return lastPractice.isEmpty() ? SheetResponseDto.from(userSheet) : SheetResponseDto.from(userSheet, lastPractice.get(0).getCreatedDate());
    }
    /**
     * 사용자 이메일 기반으로 악보 일괄 삭제
     * @param userEmail 사용자 이메일
     * @param userSheetIdList 악보 id 목록
     * @throws DataNotFoundException 요청의 악보 id에 해당하는 악보가 없다면 예외를 던지고 롤백
     * @throws InvalidRequestException 악보 주인과 요청의 이메일이 불일치하면 예외를 던짐
    * */
    @Transactional
    public void deleteSheetByIdList(String userEmail, List<Integer> userSheetIdList){
        for(Integer userSheetId : userSheetIdList){
            UserSheet sheet = userSheetRepository.findById(userSheetId)
                    .orElseThrow(() -> new DataNotFoundException("Sheet Not Found"));
            if(!sheet.getUserEmail().equals(userEmail)){
                throw new InvalidRequestException("Access Denied. You are not allowed to delete this sheet");
            }
            sheetPracticeRepository.deletePracticeByUserSheetId(userSheetId);
            userSheetRepository.deleteById(userSheetId);
        }
    }
}
