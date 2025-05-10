package com.capstone.sheet.service;

import com.capstone.data.TestDataGenerator;
import com.capstone.sheet.dto.SheetCreateMeta;
import com.capstone.sheet.dto.SheetResponseDto;
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
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.when;
import static org.awaitility.Awaitility.await;

@SpringBootTest
@ActiveProfiles("test")
class SheetManageServiceTest {
    @MockBean
    SheetToXmlConverter converter;

    @Autowired
    private SheetManageService sheetManageService;

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
        testDataGenerator.generateTestData();
        sheetXmlBytes = Files.readAllBytes(sheetXmlInfo.getFile().toPath());
        testDataGenerator.generateTestData();
        testDataGenerator.generateTestData();
    }

    @AfterEach
    void tearDown() {
        testDataGenerator.deleteAllTestData();
    }

    @Test
    void saveSheetAndUserSheet_success() {
        // given
        SheetCreateMeta meta = SheetCreateMeta.builder()
                .sheetName(UUID.randomUUID().toString())
                .color("#ffffffff")
                .fileExtension("pdf")
                .isOwner(true)
                .userEmail("test@test.com").build();
        // stub
        when(converter.convertToXml(meta, sheetFilePDF)).thenReturn(sheetXmlBytes);
        // when
        SheetResponseDto res = sheetManageService.saveSheetAndUserSheet(meta, sheetFilePDF);
        // then
        assert res.getSheetName().equals(meta.getSheetName());
        await().atMost(1, TimeUnit.MINUTES).untilAsserted(() -> {
            List<UserSheet> userSheets = userSheetRepository.findAllBySheetName(res.getSheetName());
            assertEquals(1, userSheets.size());
            assertNotNull(userSheets.get(0).getSheet().getSheetInfo());
            assertTrue(userSheets.get(0).getSheet().getSheetInfo().length > 0);
        });
    }
}