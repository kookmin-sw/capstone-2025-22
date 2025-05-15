package com.capstone.practice.service;

import com.capstone.dto.sheet.MusicServiceClientDto.PatternPracticeCreateRequest;
import com.capstone.practice.dto.PatternPracticeResponseDto;
import com.capstone.practice.entity.PatternPractice;
import com.capstone.practice.repository.PatternPracticeRepository;
import com.capstone.sheet.entity.Pattern;
import com.capstone.sheet.repository.PatternRepository;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

@Slf4j
@Service
@RequiredArgsConstructor
public class PatternPracticeManageService {

    private final PatternRepository patternRepository;
    private final PatternPracticeRepository patternPracticeRepository;
    private final ObjectMapper objectMapper;

    public PatternPracticeResponseDto savePractice(PatternPracticeCreateRequest createDto) {
        try {
            Pattern pattern = patternRepository.findById(createDto.getPatternId())
                    .orElseThrow(() -> new RuntimeException("Pattern Not Found"));
            String practiceInfo = objectMapper.writeValueAsString(createDto.getFinalMeasures());
            PatternPractice patternPractice = patternPracticeRepository.save(PatternPractice.builder()
                    .practiceInfo(practiceInfo)
                    .pattern(pattern)
                    .score(Double.toString(createDto.getScore()))
                    .userEmail(createDto.getUserEmail()).build());
            return PatternPracticeResponseDto.from(patternPractice);
        }catch (JsonProcessingException e){
            log.error(e.getMessage(), e);
            throw new RuntimeException("Json Mapping Error");
        }
    }

    public void deletePractice(int practiceId) {
        PatternPractice patternPractice = patternPracticeRepository.findById(practiceId)
                .orElseThrow(() -> new RuntimeException("PatternPractice Not Found"));
        patternPracticeRepository.delete(patternPractice);
    }
}
