package com.capstone.data;

import com.capstone.practice.entity.SheetPractice;
import com.capstone.practice.repository.SheetPracticeRepository;
import com.capstone.sheet.entity.Sheet;
import com.capstone.sheet.entity.UserSheet;
import com.capstone.sheet.repository.SheetRepository;
import com.capstone.sheet.repository.UserSheetRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;

@Component
public class TestDataGenerator {
    @Autowired
    private SheetRepository sheetRepository;
    @Autowired
    private UserSheetRepository userSheetRepository;
    @Autowired
    private SheetPracticeRepository sheetPracticeRepository;

    public static int userSheetsPerUser = 100;
    public static int sheetPracticesPerUser = 100;
    public static int sheetPracticesPerUserSheet = 10;

    public static final List<String> userEmails
            = List.of("test1@test.com", "test2@test.com", "test3@test.com");

    @Transactional
    public void generateTestData() {
        for(String userEmail : userEmails) {
            List<Sheet> sheets = generateTestSheets(userEmail);
            List<UserSheet> userSheets = generateTestUserSheets(userEmail, sheets);
            List<SheetPractice> sheetPractices = generateTestSheetPractices(userEmail, userSheets);
        }
    }
    @Transactional
    public void deleteAllTestData() {
        sheetPracticeRepository.deleteAll();
        userSheetRepository.deleteAll();
        sheetRepository.deleteAll();
    }
    @Transactional
    public List<Sheet> generateTestSheets(String email){
        List<Sheet> res = new ArrayList<>(List.of());
        for(int i=0; i<10; i++) {
            Sheet sheet = sheetRepository.save(Sheet.builder()
                    .sheetInfo("sheetInfo")
                    .createdDate(LocalDateTime.now()).build());
            res.add(sheet);
        }
        return res;
    }
    @Transactional
    public List<UserSheet> generateTestUserSheets(String email, List<Sheet> sheets){
        List<UserSheet> res = new ArrayList<>(List.of());
        for(Sheet sheet : sheets) {
            for(int i=0; i<10; i++) {
                UserSheet userSheet = userSheetRepository.save(UserSheet.builder()
                        .sheetName("sheetName")
                        .sheet(sheet)
                        .color("color")
                        .isOwner(true)
                        .userEmail(email)
                        .createdDate(LocalDateTime.now()).build());
                res.add(userSheet);
            }
        }
        return res;
    }
    @Transactional
    public List<SheetPractice> generateTestSheetPractices(String email, List<UserSheet> userSheets){
        List<SheetPractice> res = new ArrayList<>(List.of());
        for(UserSheet userSheet : userSheets) {
            for(int i=0; i<10; i++) {
                SheetPractice practice = sheetPracticeRepository.save(
                        SheetPractice.builder()
                                .practiceInfo("practiceInfo")
                                .userEmail(email)
                                .score(90)
                                .createdDate(LocalDateTime.now())
                                .userSheet(userSheet).build()
                );
                res.add(practice);
            }
        }
        return res;
    }
}
