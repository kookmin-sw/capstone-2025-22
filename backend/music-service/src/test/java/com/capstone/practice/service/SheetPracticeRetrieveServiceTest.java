package com.capstone.practice.service;

import com.capstone.data.TestDataGenerator;
import com.capstone.exception.DataNotFoundException;
import com.capstone.practice.dto.SheetPracticeDetailResponseDto;
import com.capstone.practice.dto.SheetPracticeRepresentResponse;
import com.capstone.practice.dto.SheetPracticeResponseDto;
import com.capstone.practice.entity.SheetPractice;
import com.capstone.practice.repository.SheetPracticeRepository;
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
class SheetPracticeRetrieveServiceTest {
    @Autowired
    TestDataGenerator testDataGenerator;

    @Autowired
    SheetPracticeRetrieveService sheetPracticeRetrieveService;

    @Autowired
    UserSheetRepository userSheetRepository;

    @Autowired
    SheetPracticeRepository sheetPracticeRepository;

    @BeforeEach
    void setUp() {
        testDataGenerator.generateTestData();
    }

    @AfterEach
    void tearDown() {
        testDataGenerator.deleteAllTestData();
    }

    @Test
    void getSheetPracticeRecords() {
        // given
        String email = TestDataGenerator.userEmails.get(0);
        String ghostEmail = UUID.randomUUID().toString();
        UserSheet userSheet = userSheetRepository.findAllByEmail(email).get(0);
        int pageNumber = 0;
        int overBoundPageSize = Integer.MAX_VALUE;
        int pageSize = 3;
        // when
        List<SheetPracticeResponseDto> res = sheetPracticeRetrieveService.getSheetPracticeRecords(
                email,
                pageNumber,
                pageSize,
                userSheet.getUserSheetId());
        List<SheetPracticeResponseDto> lengthMustBeTen = sheetPracticeRetrieveService.getSheetPracticeRecords(
                email,
                pageNumber,
                overBoundPageSize,
                userSheet.getUserSheetId());
        List<SheetPracticeResponseDto> mustBeZero = sheetPracticeRetrieveService.getSheetPracticeRecords(
                ghostEmail,
                pageNumber,
                pageSize,
                userSheet.getUserSheetId());
        // then
        assert res.size()==pageSize;
        assert lengthMustBeTen.size()==TestDataGenerator.sheetPracticesPerUserSheet;
        assert mustBeZero.isEmpty();
    }

    @Test
    void getDetailSheetPracticeRecord() {
        // given
        String email = TestDataGenerator.userEmails.get(0);
        UserSheet userSheet = userSheetRepository.findAllByEmail(email).get(0);
        SheetPracticeResponseDto sheetPractice = sheetPracticeRetrieveService.getSheetPracticeRecords(
                email,
                0,
                1,
                userSheet.getUserSheetId()).get(0);
        // when
        SheetPracticeDetailResponseDto res = sheetPracticeRetrieveService.getDetailSheetPracticeRecord(sheetPractice.getPracticeId());
        // then
        assert res.getSheetId()==userSheet.getUserSheetId();
        assert res.getPracticeId()==sheetPractice.getPracticeId();
    }

    @Test
    void getSheetPresentRepresentRecordTest(){
        // given
        String email = TestDataGenerator.userEmails.get(0);
        int ghostUserSheetId = -1;
        UserSheet userSheet = userSheetRepository.findAllByEmail(email).get(0);
        sheetPracticeRepository.save(SheetPractice.builder()
                .userSheet(userSheet)
                .score(Integer.MAX_VALUE)
                .userEmail(email)
                .practiceInfo("info")
                .build());
        // when
        SheetPracticeRepresentResponse res = sheetPracticeRetrieveService.getRepresentSheetPractice(email, userSheet.getUserSheetId());
        // then
        assert res.getSheetName().equals(userSheet.getSheetName());
        assert res.getMaxScore() == Integer.MAX_VALUE;
        assert res.getUserSheetId() == userSheet.getUserSheetId();
        assertThrows(DataNotFoundException.class, () -> {
           sheetPracticeRetrieveService.getRepresentSheetPractice(email, ghostUserSheetId);
        });
    }
}