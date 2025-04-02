package com.capstone.sheet.service;

import com.capstone.data.TestDataGenerator;
import com.capstone.exception.DataNotFoundException;
import com.capstone.sheet.dto.SheetDetailResponseDto;
import com.capstone.sheet.dto.SheetResponseDto;
import com.capstone.sheet.entity.UserSheet;
import com.capstone.sheet.repository.UserSheetRepository;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.*;

@SpringBootTest
@Transactional
@ActiveProfiles("test")
class SheetRetrieveServiceTest {
    @Autowired
    private UserSheetRepository userSheetRepository;

    @Autowired
    private SheetRetrieveService sheetRetrieveService;

    @Autowired
    private TestDataGenerator testDataGenerator;

    @BeforeEach
    void setUp() {
        testDataGenerator.generateTestData();
    }

    @AfterEach
    void tearDown() {
        testDataGenerator.deleteAllTestData();
    }

    @Test
    void getSheetsByEmailTest() {
        // given
        String email = TestDataGenerator.userEmails.get(0);
        String ghostUserEmail = UUID.randomUUID() + "@test.com";
        // when
        List<SheetResponseDto> res = sheetRetrieveService.getSheetsByEmail(email);
        List<SheetResponseDto> mustBeEmpty = sheetRetrieveService.getSheetsByEmail(ghostUserEmail);
        // then
        assert res.size() == TestDataGenerator.userSheetsPerUser;
        assert mustBeEmpty.isEmpty();
        assertTrue(() -> {
            for (SheetResponseDto r : res) {
                UserSheet userSheet = userSheetRepository.findById(r.getUserSheetId()).orElse(null);
                if(userSheet == null || !userSheet.getUserEmail().equals(email)) return false;
            }
            return true;
        });
    }
    @Test
    void getSheetByIdTest(){
        // given
        String email = TestDataGenerator.userEmails.get(0);
        // when
        SheetResponseDto target = sheetRetrieveService.getSheetsByEmail(email).get(0);
        SheetDetailResponseDto res = sheetRetrieveService.getSheetById(target.getUserSheetId());
        // then
        assert res != null;
        assert res.getSheetName().equals(target.getSheetName());
        assert res.getUserSheetId() == target.getUserSheetId();
        assertThrows(DataNotFoundException.class, () -> sheetRetrieveService.getSheetById(-1));
    }
}