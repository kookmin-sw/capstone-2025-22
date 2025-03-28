package com.capstone.sheet.service;

import com.capstone.sheet.dto.SheetResponseDto;
import com.capstone.sheet.entity.Sheet;
import com.capstone.sheet.entity.UserSheet;
import com.capstone.sheet.repository.SheetRepository;
import com.capstone.sheet.repository.UserSheetRepository;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.jdbc.AutoConfigureTestDatabase;
import org.springframework.boot.test.autoconfigure.orm.jpa.DataJpaTest;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.test.context.TestPropertySource;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.*;

@SpringBootTest
@Transactional
@ActiveProfiles("test")
class SheetRetrieveServiceTest {
    @Autowired
    private SheetRepository sheetRepository;

    @Autowired
    private UserSheetRepository userSheetRepository;

    @Autowired
    private SheetRetrieveService sheetRetrieveService;

    String userEmail = "test@test.com";
    @BeforeEach
    void setUp() {
        Sheet sheet = sheetRepository.save(
                Sheet.builder()
                        .sheetInfo("sheetInfo")
                        .build()
        );
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
    @AfterEach
    void tearDown() {
        userSheetRepository.deleteAll();
        sheetRepository.deleteAll();
    }
    @Test
    void getSheetsByEmail() {
        // given
        String email = userEmail;
        String ghostUserEmail = UUID.randomUUID().toString() + "@test.com";
        // when
        List<SheetResponseDto> res = sheetRetrieveService.getSheetsByEmail(email);
        List<SheetResponseDto> mustBeEmpty = sheetRetrieveService.getSheetsByEmail(ghostUserEmail);
        // then
        assert res.size() == 10;
        assert mustBeEmpty.isEmpty();
        assertTrue(() -> {
            for (SheetResponseDto r : res) {
                UserSheet userSheet = userSheetRepository.findById(r.getUserSheetId()).orElse(null);
                if(userSheet == null || !userSheet.getUserEmail().equals(email)) return false;
            }
            return true;
        });
    }
}