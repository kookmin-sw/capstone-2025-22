package com.capstone.sheet.service;

import com.capstone.sheet.dto.SheetCreateMeta;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.springframework.core.io.DefaultResourceLoader;
import org.springframework.core.io.ResourceLoader;
import org.springframework.mock.web.MockMultipartFile;
import org.springframework.test.context.junit.jupiter.SpringExtension;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;

import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(SpringExtension.class)
class SheetToXmlConverterTest {

    @InjectMocks
    private SheetToXmlConverter converter;

    ResourceLoader resourceLoader;
    MultipartFile sheetFilePDF;
    MultipartFile sheetFilePNG;

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
        input = new FileInputStream(originFile);
        sheetFilePNG = new MockMultipartFile(
                originFile.getName(),
                originFile.getName(),
                "image/png",
                input
        );
    }

    @AfterEach
    void tearDown() {
    }

    @Test
    void convertToXml_pdf_success() {
        // given
        SheetCreateMeta sheetCreateMeta = SheetCreateMeta.builder()
                .sheetName("sheet name")
                .color("#ffffffff")
                .isOwner(true)
                .userEmail("test@test.com")
                .fileExtension("pdf")
                .build();
        MultipartFile sheetFile = this.sheetFilePDF;
        // when
        byte[] res = converter.convertToXml(sheetCreateMeta, sheetFile);
        // then
        assertNotNull(res);
        assertTrue(res.length > 0);
    }
}