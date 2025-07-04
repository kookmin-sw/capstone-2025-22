package com.capstone.config;

import com.capstone.dto.musicXml.MeasureInfo;
import com.capstone.dto.musicXml.PartInfo;
import com.capstone.dto.score.FinalMeasureResult;
import com.capstone.practice.entity.SheetPractice;
import com.capstone.practice.repository.SheetPracticeRepository;
import com.capstone.sheet.entity.Sheet;
import com.capstone.sheet.entity.UserSheet;
import com.capstone.sheet.repository.SheetRepository;
import com.capstone.sheet.repository.UserSheetRepository;
import com.capstone.sheet.service.SheetXmlInfoParser;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.github.javafaker.Faker;
import jakarta.annotation.PostConstruct;
import lombok.RequiredArgsConstructor;
import org.springframework.context.annotation.Profile;
import org.springframework.core.io.DefaultResourceLoader;
import org.springframework.stereotype.Component;

import java.io.InputStream;
import java.time.LocalDateTime;
import java.util.*;

@Component
@Profile("dev")
@RequiredArgsConstructor
public class FakeSheetDataGenerator {
    private final SheetRepository sheetRepository;
    private final UserSheetRepository userSheetRepository;
    private final SheetPracticeRepository sheetPracticeRepository;
    private final SheetXmlInfoParser sheetXmlInfoParser;
    private final ObjectMapper objectMapper;

    @PostConstruct
    public void init() throws Exception {
        Faker faker = new Faker(Locale.KOREAN);
        if(sheetPracticeRepository.findAll().size() > 10){
            return;
        }
        byte[] sheetXml;
        try (InputStream inputStream = new DefaultResourceLoader()
                .getResource("classpath:sheets/sheet.xml")
                .getInputStream()) {
            sheetXml = inputStream.readAllBytes();
        }
        List<PartInfo> sheetPureJson = sheetXmlInfoParser.parseXmlInfo(sheetXml);
        String practiceJson = getPracticeInfo(sheetPureJson);
        String sheetJson = objectMapper.writeValueAsString(sheetPureJson);
        String testUser = "test@test.com";
        List<Sheet> sheets = generateSheets(faker, sheetXml, sheetJson, 10);
        List<UserSheet> userSheets = generateUserSheets(faker, sheets, testUser);
        generateSheetPractices(faker, userSheets, practiceJson,10, testUser);
    }
    /**
     * generate fake data for sheets
    * */
    public List<Sheet> generateSheets(Faker faker, byte[] sheetXml, String sheetJson, int count) {
        List<Sheet> sheets = new ArrayList<>();
        for (int i=0; i<count; i++){
            Sheet sheet = sheetRepository.save(
                    Sheet.builder()
                            .author(faker.artist().name())
                            .sheetInfo(sheetXml)
                            .sheetJson(sheetJson)
                            .build()
            );
            sheets.add(sheet);
        }
        return sheets;
    }
    /**
     * generate fake data for userSheets (user : fake user)
     * */
    public List<UserSheet> generateUserSheets(Faker faker, List<Sheet> sheets, String testUser) {
        List<UserSheet> userSheets = new ArrayList<>();
        for (Sheet sheet: sheets){
            UserSheet userSheet = userSheetRepository.save(
                    UserSheet.builder()
                            .sheetName(faker.harryPotter().book())
                            .isOwner(true)
                            .color(faker.color().hex())
                            .createdDate(LocalDateTime.now())
                            .userEmail(testUser)
                            .sheet(sheet)
                            .build()
            );
            userSheets.add(userSheet);
        }
        return userSheets;
    }
    /**
    * generate fake sheet practices
    * */
    public void generateSheetPractices(
            Faker faker,
            List<UserSheet> userSheets,
            String practiceJson,
            int countPerSheet,
            String testUser) throws Exception{
        for(UserSheet userSheet: userSheets){
            for(int i=0; i<countPerSheet; i++){
                sheetPracticeRepository.save(
                        SheetPractice.builder()
                                .userSheet(userSheet)
                                .score(faker.number().numberBetween(1, 100))
                                .practiceInfo(practiceJson)
                                .userEmail(testUser)
                                .createdDate(LocalDateTime.now())
                                .build()
                );
            }
        }
    }

    public List<Boolean> getRandomBooleans(int count){
        Random random = new Random();
        List<Boolean> booleans = new ArrayList<>();
        for(int i=0; i<count; i++){
            booleans.add(random.nextBoolean());
        }
        return booleans;
    }

    public String getPracticeInfo(List<PartInfo> sheetPureJson) throws Exception {
        List<FinalMeasureResult> finalMeasureResults = new ArrayList<>();
        for(PartInfo partInfo: sheetPureJson){
            for(MeasureInfo measureInfo : partInfo.getMeasureList()){
                String measureNumber = measureInfo.getMeasureNumber();
                int noteCount = measureInfo.getNoteList().size();
                int score = 0;
                List<Boolean> beatScoringResults = getRandomBooleans(noteCount);
                List<Boolean> finalScoringResults = getRandomBooleans(noteCount);
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
            }
        }
        return objectMapper.writeValueAsString(finalMeasureResults);
    }
}
