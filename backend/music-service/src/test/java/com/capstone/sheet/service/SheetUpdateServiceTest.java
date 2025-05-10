package com.capstone.sheet.service;

import com.capstone.data.TestDataGenerator;
import com.capstone.exception.DataNotFoundException;
import com.capstone.exception.InvalidRequestException;
import com.capstone.sheet.dto.SheetCreateMeta;
import com.capstone.sheet.dto.SheetResponseDto;
import com.capstone.sheet.entity.Sheet;
import com.capstone.sheet.entity.UserSheet;
import com.capstone.sheet.repository.SheetRepository;
import com.capstone.sheet.repository.UserSheetRepository;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.core.io.DefaultResourceLoader;
import org.springframework.core.io.Resource;
import org.springframework.core.io.ResourceLoader;
import org.springframework.mock.web.MockMultipartFile;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.util.List;
import java.util.UUID;

import static java.lang.Thread.sleep;
import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.when;

@SpringBootTest
@Transactional
@ActiveProfiles("test")
class SheetUpdateServiceTest {
    @MockBean
    SheetToXmlConverter converter;

    @Autowired
    private SheetUpdateService sheetUpdateService;

    @Autowired
    private SheetCreateService sheetCreateService;

    @Autowired
    private SheetRetrieveService sheetRetrieveService;

    @Autowired
    private SheetRepository sheetRepository;

    @Autowired
    private UserSheetRepository userSheetRepository;

    @Autowired
    private TestDataGenerator testDataGenerator;

    ResourceLoader resourceLoader;
    MultipartFile sheetFilePDF;
    byte[] sheetXmlBytes;

    @BeforeEach
    void setUp() throws IOException {
        resourceLoader = new DefaultResourceLoader();
        File originFile = resourceLoader.getResource("classpath:sheet/sheet.pdf").getFile();
        FileInputStream input = new FileInputStream(originFile);
        sheetFilePDF = new MockMultipartFile(
                originFile.getName(),   // file name
                originFile.getName(),   // origin file name
                "application/pdf",      // content type
                input                   // input
        );
        Resource sheetXmlInfo = resourceLoader.getResource("classpath:sheet/sheet.xml");
        sheetXmlBytes = Files.readAllBytes(sheetXmlInfo.getFile().toPath());
        testDataGenerator.generateTestData();
    }

    @AfterEach
    void tearDown() {
        testDataGenerator.deleteAllTestData();
    }

    @Test
    void updateSheetInfoTest()  throws InterruptedException, IOException{
        // given
        Sheet sheet = sheetCreateService.saveSheet();
        SheetCreateMeta meta = SheetCreateMeta.builder()
                .sheetName("sheetName")
                .color("#ffffffff")
                .fileExtension("pdf")
                .isOwner(true)
                .userEmail("test@test.com").build();

        // stub
        when(converter.convertToXml(meta, sheetFilePDF)).thenReturn(sheetXmlBytes);

        // when
        sheetUpdateService.updateSheetInfo(sheet, meta, sheetFilePDF);

        // then
        assertTrue(() -> {
            Sheet updatedSheet = sheetRepository.findById(sheet.getSheetId()).orElse(null);
            if(updatedSheet==null) return false;
            return updatedSheet.getSheetInfo() != null;
        });
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
        SheetResponseDto res = sheetUpdateService.updateSheetName(email, newName, userSheet.getUserSheetId());
        // then
        assert res.getSheetName().equals(newName);
        assertThrows(InvalidRequestException.class, () -> {
            sheetUpdateService.updateSheetName(ghostEmail, newName, userSheet.getUserSheetId());
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
        SheetResponseDto res = sheetUpdateService.updateSheetColor(email, newColor, userSheet.getUserSheetId());
        // then
        assert res.getColor().equals(newColor);
        assertThrows(InvalidRequestException.class, () -> {
            sheetUpdateService.updateSheetName(ghostEmail, newColor, userSheet.getUserSheetId());
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
            sheetUpdateService.deleteSheetByIdList(email, List.of(-1));
        });
        assertThrows(InvalidRequestException.class, () -> {
            sheetUpdateService.deleteSheetByIdList(ghostEmail, userSheetIds);
        });
        assertTrue(() -> {
            if(sheetRetrieveService.getSheetsByEmail(email).size()!=userSheetIds.size()) return false;
            sheetUpdateService.deleteSheetByIdList(email, userSheetIds);
            return sheetRetrieveService.getSheetsByEmail(email).isEmpty();
        });
    }
}