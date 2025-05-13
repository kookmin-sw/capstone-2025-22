package com.capstone.practice.service;

import com.capstone.practice.dto.PatternPracticeResponseDto;
import com.capstone.practice.repository.PatternPracticeRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
@RequiredArgsConstructor
public class PatternPracticeRetrieveService {

    private final PatternPracticeRepository patternPracticeRepository;

    public List<PatternPracticeResponseDto> getUserPracticeRecordsByPattern(String userEmail, Long patternId){
        return patternPracticeRepository.findByPatternAndUserEmail(patternId, userEmail)
                .stream()
                .map(PatternPracticeResponseDto::from).toList();
    }
}
