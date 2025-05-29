package com.capstone.sheet.service;

import com.capstone.exception.DataNotFoundException;
import com.capstone.exception.InternalServerException;
import com.capstone.sheet.dto.PatternCreateDto;
import com.capstone.sheet.dto.PatternResponseDto;
import com.capstone.sheet.dto.PatternUpdateDto;
import com.capstone.sheet.entity.Pattern;
import com.capstone.sheet.repository.PatternRepository;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;

@Slf4j
@Service
@RequiredArgsConstructor
public class PatternManageService {

    private final PatternRepository patternRepository;

    private final SheetToXmlConverter sheetToXmlConverter;

    private final SheetXmlInfoParser sheetXmlInfoParser;

    private final ObjectMapper objectMapper;

    @Transactional
    public PatternResponseDto savePattern(PatternCreateDto createDto, byte[] sheetFile, byte[] patternWav){
        try{
            byte[] sheetXml = sheetToXmlConverter.convertToXml(createDto, sheetFile);
            String sheetJson = objectMapper.writeValueAsString(sheetXmlInfoParser.parseXmlInfo(sheetXml));
            Pattern pattern = patternRepository.save(Pattern.builder()
                    .patternJson(sheetJson)
                    .patternInfo(sheetXml)
                    .patternWav(patternWav)
                    .patternName(createDto.getPatternName()).build());
            return PatternResponseDto.from(pattern);
        }catch (Exception e){
            log.error("Exception occurred while saving pattern", e);
            throw new InternalServerException("Exception occurred while saving pattern : " + e.getMessage());
        }
    }

    @Transactional
    public PatternResponseDto updatePatternInfo(Long patternId, PatternUpdateDto patternUpdateDto, byte[] sheetFile, byte[] patternWav){
        String fileExtension = patternUpdateDto.getFileExtension();
        Pattern pattern = patternRepository.findById(patternId).orElseThrow(() -> new DataNotFoundException("Pattern Not Found"));
        if(patternUpdateDto.getPatternName()!=null){
            pattern.setPatternName(patternUpdateDto.getPatternName());
        }
        if(sheetFile==null || patternWav==null){
            return PatternResponseDto.from(pattern);
        }
        try {
            PatternCreateDto patternCreateDto = PatternCreateDto.builder()
                    .fileExtension(fileExtension)
                    .patternName(pattern.getPatternName())
                    .build();
            byte[] sheetXml = sheetToXmlConverter.convertToXml(patternCreateDto, sheetFile);
            String sheetJson = objectMapper.writeValueAsString(sheetXmlInfoParser.parseXmlInfo(sheetXml));
            pattern.setPatternJson(sheetJson);
            pattern.setPatternInfo(sheetXml);
            pattern.setPatternWav(patternWav);
            return PatternResponseDto.from(pattern);
        } catch (Exception e) {
            log.error("Exception occurred while updating pattern", e);
            throw new InternalServerException("Exception occurred while updating pattern : " + e.getMessage());
        }
    }

    @Transactional
    public void deletePattern(Long patternId){
        Pattern pattern = patternRepository.findById(patternId)
                .orElseThrow(() -> new DataNotFoundException("Pattern Not Found"));
        patternRepository.delete(pattern);
    }
}
