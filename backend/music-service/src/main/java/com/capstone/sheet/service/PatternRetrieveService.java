package com.capstone.sheet.service;

import com.capstone.dto.musicXml.MeasureInfo;
import com.capstone.dto.musicXml.PartInfo;
import com.capstone.exception.DataNotFoundException;
import com.capstone.practice.entity.PatternPractice;
import com.capstone.practice.repository.PatternPracticeRepository;
import com.capstone.sheet.dto.PatternResponseDto;
import com.capstone.sheet.entity.Pattern;
import com.capstone.sheet.repository.PatternRepository;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.ws.rs.InternalServerErrorException;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.util.List;

@Slf4j
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
        int successScore = 60;
        return patternRepository.findByUserEmailAndScoreGreaterThanEqual(userEmail, successScore)
                .stream()
                .map(pattern -> {
                    List<PatternPractice> practices = patternPracticeRepository.findMaxScorePracticesByPatternId(pattern.getId(), userEmail);
                    if(!practices.isEmpty()) return PatternResponseDto.from(pattern, Integer.parseInt(practices.get(0).getScore()));
                    else return null;
                }).toList();
    }

    public List<PatternResponseDto> getPatternsByEmail(String userEmail){
        return patternRepository.findAll().stream().map(pattern -> {
            List<PatternPractice> practices = patternPracticeRepository.findMaxScorePracticesByPatternId(pattern.getId(), userEmail);
            if(!practices.isEmpty()) return PatternResponseDto.from(pattern, Integer.parseInt(practices.get(0).getScore()));
            else return PatternResponseDto.from(pattern);
        }).toList();
    }

    public MeasureInfo findMeasureInfo(Long patternId, String measureNumber){
        try {
            Pattern pattern = patternRepository.findById(patternId)
                    .orElseThrow(() -> new DataNotFoundException("Pattern Not Found"));
            String sheetJson = pattern.getPatternJson();
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
