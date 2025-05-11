package com.capstone.sheet.service;

import com.capstone.sheet.dto.SheetCreateMeta;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.mock.mockito.SpyBean;
import org.springframework.core.io.DefaultResourceLoader;
import org.springframework.core.io.ResourceLoader;
import org.springframework.mock.web.MockMultipartFile;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.doReturn;

@SpringBootTest
@ActiveProfiles("test")
class SheetToXmlConverterTest {

    @SpyBean
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
    void convertToXml_pdf_success() throws IOException{
        // given
        SheetCreateMeta sheetCreateMeta = SheetCreateMeta.builder()
                .sheetName("sheet name")
                .color("#ffffffff")
                .isOwner(true)
                .userEmail("test@test.com")
                .fileExtension("pdf")
                .build();
        MultipartFile sheetFile = this.sheetFilePDF;
        // stub
        doReturn(new byte[100]).when(converter).processConvert(anyString(), anyString());
        // when
        byte[] res = converter.convertToXml(sheetCreateMeta, sheetFile.getBytes());
        // then
        assertNotNull(res);
        assertTrue(res.length > 0);
    }
}