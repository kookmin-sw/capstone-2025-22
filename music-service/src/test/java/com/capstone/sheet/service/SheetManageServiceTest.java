package com.capstone.sheet.service;

import com.capstone.exception.DataNotFoundException;
import com.capstone.exception.InvalidRequestException;
import com.capstone.sheet.dto.SheetResponseDto;
import com.capstone.sheet.entity.Sheet;
import com.capstone.sheet.entity.UserSheet;
import com.capstone.sheet.repository.SheetRepository;
import com.capstone.sheet.repository.UserSheetRepository;
import com.netflix.discovery.converters.Auto;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.transaction.annotation.Transactional;

import java.util.Arrays;
import java.util.List;
import java.util.UUID;
import java.util.stream.Collectors;

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
    private SheetRepository sheetRepository;

    @Autowired
    private UserSheetRepository userSheetRepository;

    List<String> userEmails = Arrays.asList("test@gmail.com", "test2@gmail.com");

    @BeforeEach
    void setUp() {
        Sheet sheet = sheetRepository.save(
                Sheet.builder()
                        .sheetInfo("sheetInfo")
                        .build()
        );
        for (String userEmail : userEmails) {
            for(int i=0; i<10; i++) {
                userSheetRepository.save(
                        UserSheet.builder()
                                .sheetName("init")
                                .color("init")
                                .userEmail(userEmail)
                                .sheet(sheet)
                                .build()
                );
            }
        }
    }

    @AfterEach
    void tearDown() {
        userSheetRepository.deleteAll();
        sheetRepository.deleteAll();
    }

    @Test
    void updateSheetNameTest() {
        // given
        String email = userEmails.get(0);
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
        String email = userEmails.get(0);
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
        String email = userEmails.get(0);
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