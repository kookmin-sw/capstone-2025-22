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
import org.springframework.stereotype.Component;

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
    public void init() {
        Faker faker = new Faker(Locale.KOREAN);
        String testUser = "test@test.com";
        List<Sheet> sheets = generateSheets(faker, 10);
        List<UserSheet> userSheets = generateUserSheets(faker, sheets, 10, testUser);
        generateSheetPractices(faker, userSheets, 10, testUser);
    }
    /**
     * generate fake data for sheets
    * */
    public List<Sheet> generateSheets(Faker faker, int count) {
        List<Sheet> sheets = new ArrayList<>();
        for (int i=0; i<count; i++){
            Sheet sheet = sheetRepository.save(
                    Sheet.builder()
                            .sheetInfo(faker.lorem().paragraph(10))
                            .build()
            );
            sheets.add(sheet);
        }
        return sheets;
    }
    /**
     * generate fake data for userSheets (user : fake user)
     * */
    public List<UserSheet> generateUserSheets(Faker faker, List<Sheet> sheets, int countPerSheet, String testUser) {
        List<UserSheet> userSheets = new ArrayList<>();
        for (int i=0; i<countPerSheet; i++){
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
        }
        return userSheets;
    }
    /**
    * generate fake sheet practices
    * */
    public void generateSheetPractices(Faker faker, List<UserSheet> userSheets, int countPerSheet, String testUser) {
        for(UserSheet userSheet: userSheets){
            sheetPracticeRepository.save(
                    SheetPractice.builder()
                            .userSheet(userSheet)
                            .score(faker.number().numberBetween(1, 100))
                            .practiceInfo(faker.lorem().paragraph(10))
                            .userEmail(testUser)
                            .createdDate(LocalDateTime.now())
                            .build()
            );
        }
    }
}
