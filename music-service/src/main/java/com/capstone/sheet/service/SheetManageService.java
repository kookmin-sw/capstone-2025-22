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

    @Transactional
    public void deleteSheetByIdList(String userEmail, List<Integer> userSheetIdList){
        for(Integer userSheetId : userSheetIdList){
            UserSheet sheet = userSheetRepository.findById(userSheetId)
                    .orElseThrow(() -> new DataNotFoundException("Sheet Not Found"));
            if(!sheet.getUserEmail().equals(userEmail)){
                throw new InvalidRequestException("Access Denied. You are not allowed to delete this sheet");
            }
            userSheetRepository.deleteById(userSheetId);
        }
    }
}
