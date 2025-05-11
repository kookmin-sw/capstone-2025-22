package com.capstone.config;

import com.capstone.dto.musicXml.PartInfo;
import com.capstone.dto.score.FinalMeasureResult;
import com.capstone.practice.entity.SheetPractice;
import com.capstone.practice.repository.SheetPracticeRepository;
import com.capstone.sheet.entity.Sheet;
import com.capstone.sheet.entity.UserSheet;
import com.capstone.sheet.repository.SheetRepository;
import com.capstone.sheet.repository.UserSheetRepository;
import com.capstone.sheet.service.SheetXmlInfoParser;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.github.javafaker.Faker;
import jakarta.annotation.PostConstruct;
import lombok.RequiredArgsConstructor;
import org.springframework.context.annotation.Profile;
import org.springframework.core.io.DefaultResourceLoader;
import org.springframework.stereotype.Component;

import java.io.InputStream;
import java.nio.file.Files;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.UUID;

@Component
@Profile("dev")
@RequiredArgsConstructor
public class FakeDataGenerator {
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
        String sheetJson = objectMapper.writeValueAsString(sheetXmlInfoParser.parseXmlInfo(sheetXml));
        String testUser = "test@test.com";
        List<Sheet> sheets = generateSheets(faker, sheetXml, sheetJson, 10);
        List<UserSheet> userSheets = generateUserSheets(faker, sheets, testUser);
        generateSheetPractices(faker, userSheets, 10, testUser);
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
    public void generateSheetPractices(Faker faker, List<UserSheet> userSheets, int countPerSheet, String testUser) throws Exception{
        FinalMeasureResult finalMeasureResult = FinalMeasureResult.builder()
                .score((double) faker.number().numberBetween(1, 100))
                .beatScoringResults(List.of(true, true, true, true, true))
                .finalScoringResults(List.of(true, true, true, true, true)).build();
        for(UserSheet userSheet: userSheets){
            for(int i=0; i<countPerSheet; i++){
                int measureNumber = 1;
                List<FinalMeasureResult> finalMeasureResults = new ArrayList<>();
                for(int j=0; j<10; j++){
                    finalMeasureResult.setMeasureNumber(Integer.toString(measureNumber++));
                    finalMeasureResults.add(finalMeasureResult);
                }
                String practiceInfo = new ObjectMapper().writeValueAsString(finalMeasureResults);
                sheetPracticeRepository.save(
                        SheetPractice.builder()
                                .userSheet(userSheet)
                                .score(faker.number().numberBetween(1, 100))
                                .practiceInfo(practiceInfo)
                                .userEmail(testUser)
                                .createdDate(LocalDateTime.now())
                                .build()
                );
            }
        }
    }
}
