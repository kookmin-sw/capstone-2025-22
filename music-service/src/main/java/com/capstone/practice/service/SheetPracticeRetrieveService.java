package com.capstone.practice.service;

import com.capstone.exception.DataNotFoundException;
import com.capstone.exception.InvalidRequestException;
import com.capstone.practice.dto.SheetPracticeDetailResponseDto;
import com.capstone.practice.dto.SheetPracticeResponseDto;
import com.capstone.practice.entity.SheetPractice;
import com.capstone.practice.repository.SheetPracticeRepository;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class SheetPracticeRetrieveService {
    private final SheetPracticeRepository sheetPracticeRepository;
    public SheetPracticeRetrieveService(SheetPracticeRepository sheetPracticeRepository) {
        this.sheetPracticeRepository = sheetPracticeRepository;
    }
    public List<SheetPracticeResponseDto> getSheetPracticeRecords(
            String userEmail,
            int pageNumber,
            int pageSize,
            int userSheetId){
        return sheetPracticeRepository
                .findAllByEmailAndSheetId(userEmail, userSheetId, PageRequest.of(pageNumber, pageSize))
                .stream().map(SheetPracticeResponseDto::from).toList();
    }
    public SheetPracticeDetailResponseDto getDetailSheetPracticeRecord(int sheetPracticeId){
        SheetPractice sheetPractice = sheetPracticeRepository.findById(sheetPracticeId)
                .orElseThrow(() -> new DataNotFoundException("SheetPractice not found"));
        return SheetPracticeDetailResponseDto.from(sheetPractice);
    }
}
