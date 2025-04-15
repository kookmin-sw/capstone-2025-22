package com.capstone.sheet.service;

import com.capstone.data.TestDataGenerator;
import com.capstone.exception.DataNotFoundException;
import com.capstone.exception.InvalidRequestException;
import com.capstone.sheet.dto.SheetCreateRequestDto;
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
class SheetManageServiceTest {
    @Autowired
    private SheetManageService sheetManageService;

    @Autowired
    private SheetRetrieveService sheetRetrieveService;

    @Autowired
    private UserSheetRepository userSheetRepository;

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
    void createSheetTest_success(){
        // given
        SheetCreateRequestDto requestDto = SheetCreateRequestDto.builder()
                .sheetName("test")
                .sheetInfo("test")
                .author("test")
                .color("#FFFFFF")
                .userEmail("test@test.com").build();
        // when
        SheetResponseDto res = sheetManageService.createSheetAndUserSheet(requestDto);
        // then
        assertNotNull(res);
        assert res.getSheetName().equals(requestDto.getSheetName());
        assert res.getAuthor().equals(requestDto.getAuthor());
        assert res.getColor().equals(requestDto.getColor());
    }

    @Test
    void updateSheetNameTest() {
        // given
        String email = TestDataGenerator.userEmails.get(0);
        String ghostEmail = UUID.randomUUID() + "@gmail.com";
        String newName = "newName"+ UUID.randomUUID();
        // when
        List<UserSheet> userSheets = userSheetRepository.findAllByEmail(email);
        UserSheet userSheet = userSheets.get(0);
        SheetResponseDto res = sheetManageService.updateSheetName(email, newName, userSheet.getUserSheetId());
        // then
        assert res.getSheetName().equals(newName);
        assertThrows(InvalidRequestException.class, () -> {
            sheetManageService.updateSheetName(ghostEmail, newName, userSheet.getUserSheetId());
        });
    }

    @Test
    void updateSheetColorTest() {
        // given
        String email = TestDataGenerator.userEmails.get(0);
        String ghostEmail = UUID.randomUUID() + "@gmail.com";
        String newColor = "newColor"+ UUID.randomUUID();
        // when
        List<UserSheet> userSheets = userSheetRepository.findAllByEmail(email);
        UserSheet userSheet = userSheets.get(0);
        SheetResponseDto res = sheetManageService.updateSheetColor(email, newColor, userSheet.getUserSheetId());
        // then
        assert res.getColor().equals(newColor);
        assertThrows(InvalidRequestException.class, () -> {
            sheetManageService.updateSheetName(ghostEmail, newColor, userSheet.getUserSheetId());
        });
    }

    @Test
    void deleteSheetsTest() {
        // given
        String email = TestDataGenerator.userEmails.get(0);
        String ghostEmail = UUID.randomUUID() + "@gmail.com";
        // when
        List<Integer> userSheetIds = sheetRetrieveService.getSheetsByEmail(email)
                .stream()
                .map(SheetResponseDto::getUserSheetId)
                .toList();
        // then
        assertThrows(DataNotFoundException.class, () -> {
            sheetManageService.deleteSheetByIdList(email, List.of(-1));
        });
        assertThrows(InvalidRequestException.class, () -> {
            sheetManageService.deleteSheetByIdList(ghostEmail, userSheetIds);
        });
        assertTrue(() -> {
            if(sheetRetrieveService.getSheetsByEmail(email).size()!=userSheetIds.size()) return false;
            sheetManageService.deleteSheetByIdList(email, userSheetIds);
            return sheetRetrieveService.getSheetsByEmail(email).isEmpty();
        });
    }
}