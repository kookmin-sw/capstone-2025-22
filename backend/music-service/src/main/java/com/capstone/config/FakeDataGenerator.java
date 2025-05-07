package com.capstone.config;

import com.capstone.practice.entity.SheetPractice;
import com.capstone.practice.repository.SheetPracticeRepository;
import com.capstone.sheet.entity.Sheet;
import com.capstone.sheet.entity.UserSheet;
import com.capstone.sheet.repository.SheetRepository;
import com.capstone.sheet.repository.UserSheetRepository;
import com.github.javafaker.Faker;
import jakarta.annotation.PostConstruct;
import org.springframework.context.annotation.Profile;
import org.springframework.core.io.DefaultResourceLoader;
import org.springframework.stereotype.Component;

import java.nio.file.Files;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

@Component
@Profile("dev")
public class FakeDataGenerator {
    private final SheetRepository sheetRepository;
    private final UserSheetRepository userSheetRepository;
    private final SheetPracticeRepository sheetPracticeRepository;
    public FakeDataGenerator(
            SheetRepository sheetRepository,
            UserSheetRepository userSheetRepository,
            SheetPracticeRepository sheetPracticeRepository) {
        this.sheetRepository = sheetRepository;
        this.userSheetRepository = userSheetRepository;
        this.sheetPracticeRepository = sheetPracticeRepository;
    }

    @PostConstruct
    public void init() throws Exception {
        Faker faker = new Faker(Locale.KOREAN);
        if(sheetRepository.count() > 10) return;
        byte[] sheetXml = Files.readAllBytes(new DefaultResourceLoader().getResource("classpath:sheets/sheet.xml")
                .getFile().toPath());
        String testUser = "test@test.com";
        List<Sheet> sheets = generateSheets(faker, sheetXml, 10);
        List<UserSheet> userSheets = generateUserSheets(faker, sheets, testUser);
        generateSheetPractices(faker, userSheets, 10, testUser);
    }
    /**
     * generate fake data for sheets
    * */
    public List<Sheet> generateSheets(Faker faker, byte[] sheetXml, int count) {
        List<Sheet> sheets = new ArrayList<>();
        for (int i=0; i<count; i++){
            Sheet sheet = sheetRepository.save(
                    Sheet.builder()
                            .author(faker.artist().name())
                            .sheetInfo(sheetXml)
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
    public void generateSheetPractices(Faker faker, List<UserSheet> userSheets, int countPerSheet, String testUser) {
        for(UserSheet userSheet: userSheets){
            for(int i=0; i<countPerSheet; i++){
                sheetPracticeRepository.save(
                        SheetPractice.builder()
                                .userSheet(userSheet)
                                .score(faker.number().numberBetween(1, 100))
                                .practiceInfo("must not use yet")
                                .userEmail(testUser)
                                .createdDate(LocalDateTime.now())
                                .build()
                );
            }
        }
    }
}
