package com.capstone.config;

import com.capstone.dto.musicXml.MeasureInfo;
import com.capstone.dto.musicXml.PartInfo;
import com.capstone.dto.score.FinalMeasureResult;
import com.capstone.practice.entity.PatternPractice;
import com.capstone.practice.repository.PatternPracticeRepository;
import com.capstone.practice.repository.SheetPracticeRepository;
import com.capstone.sheet.dto.PatternCreateDto;
import com.capstone.sheet.entity.Pattern;
import com.capstone.sheet.repository.PatternRepository;
import com.capstone.sheet.service.SheetToXmlConverter;
import com.capstone.sheet.service.SheetXmlInfoParser;
import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.annotation.PostConstruct;
import lombok.RequiredArgsConstructor;
import org.springframework.context.annotation.Profile;
import org.springframework.core.io.DefaultResourceLoader;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

@Component
@Profile("dev")
@RequiredArgsConstructor
public class FakePatternDataGenerator {

    private final PatternPracticeRepository patternPracticeRepository;
    private final PatternRepository patternRepository;
    private final SheetXmlInfoParser sheetXmlInfoParser;
    private final SheetToXmlConverter sheetToXmlConverter;
    private final ObjectMapper objectMapper;

    @PostConstruct
    public void init() {
        if(patternRepository.findAll().size() >= 10){
            return;
        }
        try {
            var patternSheet = new DefaultResourceLoader().getResource("/patterns/test_pattern.pdf");
            var patternWav = new DefaultResourceLoader().getResource("/patterns/test_pattern.wav");
            PatternCreateDto patternCreateDto = PatternCreateDto.builder()
                    .patternName("test pattern")
                    .fileExtension("pdf")
                    .build();
            byte[] patternSheetBytes = patternSheet.getInputStream().readAllBytes();
            byte[] patternWavBytes = patternWav.getInputStream().readAllBytes();
            byte[] patternXmlBytes = sheetToXmlConverter.convertToXml(patternCreateDto, patternSheetBytes);
            patternXmlBytes = sheetToXmlConverter.cleanMusicXml(patternXmlBytes);
            List<PartInfo> partInfoList = sheetXmlInfoParser.parseXmlInfo(patternXmlBytes);
            String patternJson = objectMapper.writeValueAsString(partInfoList);
            List<Pattern> patternList = createTestPattern(patternWavBytes, patternXmlBytes, patternJson, 10);
            List<PatternPractice> patternPractices = createTestPatternPractices(partInfoList, patternList, 10);
        }catch (Exception e){
            throw new RuntimeException("Exception occurred while generating fake pattern data", e);
        }
    }

    public List<Pattern> createTestPattern(byte[] patternWavBytes, byte[] patternXmlBytes, String patternJson, int size) {
        List<Pattern> savedPatterns = new ArrayList<>();
        Pattern.PatternBuilder patternBuilder = Pattern.builder()
                .patternInfo(patternXmlBytes)
                .patternJson(patternJson)
                .patternWav(patternWavBytes);
        for(int i=0; i<size; i++){
            String patternName = "test pattern " + (i+1);
            Pattern pattern = patternBuilder.patternName(patternName).build();
            savedPatterns.add(patternRepository.save(pattern));
        }
        return savedPatterns;
    }

    public List<Boolean> getRandomBooleans(int count, boolean mustBeTrue){
        Random random = new Random();
        List<Boolean> booleans = new ArrayList<>();
        for(int i=0; i<count; i++){
            if(!mustBeTrue) booleans.add(random.nextBoolean());
            else booleans.add(true);
        }
        return booleans;
    }

    public List<PatternPractice> createTestPatternPractices(List<PartInfo> patternPureJson, List<Pattern> patterns, int size) throws Exception {
        List<FinalMeasureResult> finalMeasureResults = new ArrayList<>();
        double totalScore = 0;
        int measureCount = patternPureJson.get(0).getMeasureList().size();
        for(PartInfo partInfo: patternPureJson){
            for(MeasureInfo measureInfo : partInfo.getMeasureList()){
                String measureNumber = measureInfo.getMeasureNumber();
                int noteCount = measureInfo.getNoteList().size();
                int score = 0;
                List<Boolean> beatScoringResults = getRandomBooleans(noteCount, true);
                List<Boolean> finalScoringResults = getRandomBooleans(noteCount, true);
                for(int i=0; i<noteCount; i++){
                    double resScore = 0;
                    double unitScore = (double) 100 / noteCount;
                    if(beatScoringResults.get(i)) resScore += (unitScore * 0.7);
                    if(finalScoringResults.get(i)) resScore += (unitScore * 0.3);
                    score += (int) resScore;
                }
                FinalMeasureResult finalMeasureResult = FinalMeasureResult.builder()
                        .beatScoringResults(beatScoringResults)
                        .finalScoringResults(finalScoringResults)
                        .score((double) score)
                        .measureNumber(measureNumber)
                        .build();
                finalMeasureResults.add(finalMeasureResult);
                totalScore += score;
            }
        }
        String practiceInfo = objectMapper.writeValueAsString(finalMeasureResults);
        PatternPractice.PatternPracticeBuilder patternPracticeBuilder = PatternPractice.builder()
                .practiceInfo(practiceInfo)
                .score(Integer.toString((int) totalScore / measureCount))
                .userEmail("test@test.com");
        List<PatternPractice> savedPractices = new ArrayList<>();
        for(Pattern pattern: patterns){
            for(int i=0; i<size; i++){
                PatternPractice patternPractice = patternPracticeBuilder.pattern(pattern).build();
                savedPractices.add(patternPracticeRepository.save(patternPractice));
            }
        }
        return savedPractices;
    }
}
