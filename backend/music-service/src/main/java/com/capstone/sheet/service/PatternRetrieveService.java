package com.capstone.sheet.service;

import com.capstone.exception.DataNotFoundException;
import com.capstone.practice.repository.PatternPracticeRepository;
import com.capstone.sheet.dto.PatternResponseDto;
import com.capstone.sheet.entity.Pattern;
import com.capstone.sheet.repository.PatternRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
@RequiredArgsConstructor
public class PatternRetrieveService {
    private final PatternRepository patternRepository;
    private final PatternPracticeRepository patternPracticeRepository;

    public List<PatternResponseDto> getAllPatterns(){
        return patternRepository.findAll()
                .stream()
                .map(PatternResponseDto::from)
                .toList();
    }

    public PatternResponseDto getPatternById(Long patternId){
        Pattern pattern = patternRepository.findById(patternId)
                .orElseThrow(() -> new DataNotFoundException("Pattern Not Exists"));
        return PatternResponseDto.from(pattern);
    }

    public List<PatternResponseDto> getSucceededPatternsByEmail(String userEmail){
        int successScore = 80;
        return patternRepository.findByUserEmailAndScoreGreaterThanEqual(userEmail, successScore)
                .stream()
                .map(PatternResponseDto::from)
                .toList();
    }
}
