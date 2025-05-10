package com.capstone.sheet.controller;

import com.capstone.data.TestDataGenerator;
import com.capstone.enums.SuccessFlag;
import com.capstone.practice.entity.SheetPractice;
import com.capstone.practice.repository.SheetPracticeRepository;
import com.capstone.sheet.dto.SheetCreateMeta;
import com.capstone.sheet.dto.SheetListRequestDto;
import com.capstone.sheet.dto.SheetUpdateRequestDto;
import com.capstone.sheet.entity.UserSheet;
import com.capstone.sheet.repository.UserSheetRepository;
import com.capstone.sheet.service.SheetToXmlConverter;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.core.io.DefaultResourceLoader;
import org.springframework.core.io.Resource;
import org.springframework.core.io.ResourceLoader;
import org.springframework.data.domain.PageRequest;
import org.springframework.http.MediaType;
import org.springframework.mock.web.MockMultipartFile;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.test.web.servlet.MockMvc;

import java.io.File;
import java.io.FileInputStream;
import java.nio.file.Files;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.*;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.*;
import static org.springframework.test.web.servlet.result.MockMvcResultHandlers.print;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;
import static org.awaitility.Awaitility.await;
import static org.mockito.Mockito.when;

@SpringBootTest
@ActiveProfiles("test")
@AutoConfigureMockMvc
class SheetManageControllerTest {
    @MockBean
    SheetToXmlConverter converter;

    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private ObjectMapper objectMapper;

    @Autowired
    private UserSheetRepository userSheetRepository;

    @Autowired
    private SheetPracticeRepository sheetPracticeRepository;

    @Autowired
    TestDataGenerator testDataGenerator;

    byte[] sheetXmlBytes;

    @BeforeEach
    void setUp() throws Exception{
        ResourceLoader resourceLoader = new DefaultResourceLoader();
        Resource sheetXmlInfo = resourceLoader.getResource("classpath:sheet/sheet.xml");
        testDataGenerator.generateTestData();
        sheetXmlBytes = Files.readAllBytes(sheetXmlInfo.getFile().toPath());
        testDataGenerator.generateTestData();
    }

    @AfterEach
    void tearDown() {
        testDataGenerator.deleteAllTestData();
    }

    @Test
    @DisplayName("악보 정보 저장 성공 테스트")
    void createSheet_success() throws Exception {
        // given
        String email = TestDataGenerator.userEmails.get(0);
        String sheetName = UUID.randomUUID().toString();
        File originFile = new DefaultResourceLoader().getResource("classpath:sheet/sheet.pdf").getFile();
        SheetCreateMeta meta = SheetCreateMeta.builder()
                .sheetName(sheetName)
                .userEmail(email)
                .isOwner(true)
                .fileExtension("pdf")
                .color("#ffffffff")
                .build();
        MockMultipartFile sheetFilePDF = new MockMultipartFile(
                "sheetFile",
                originFile.getName(),
                "application/pdf",
                new FileInputStream(originFile)                   // input
        );
        MockMultipartFile metaPart = new MockMultipartFile(
                "sheetCreateMeta",
                "",
                "application/json",
                new ObjectMapper().writeValueAsString(meta).getBytes()
        );
        // stub
        when(converter.convertToXml(meta, sheetFilePDF)).thenReturn(sheetXmlBytes);

        // when & then
        mockMvc.perform(multipart("/sheets")
                        .file(sheetFilePDF)
                        .file(metaPart))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.body.sheetName").value(sheetName))
                .andDo(print());
        await().atMost(1, TimeUnit.MINUTES).untilAsserted(() -> {
            List<UserSheet> userSheets = userSheetRepository.findAllBySheetName(sheetName);
            assertEquals(1, userSheets.size());
            assertNotNull(userSheets.get(0).getSheet().getSheetInfo());
            assertTrue(userSheets.get(0).getSheet().getSheetInfo().length > 0);
        });
    }

    @Test
    @DisplayName("이름 변경 성공 테스트")
    void updateSheetName_success() throws Exception {
        // given
        UserSheet targetSheet = userSheetRepository.findAll().get(0);
        String newName = UUID.randomUUID().toString();
        SheetUpdateRequestDto requestDto = SheetUpdateRequestDto.builder()
                .email(targetSheet.getUserEmail())
                .name(newName).build();
        String body = objectMapper.writeValueAsString(requestDto);
        // when & then
        mockMvc.perform(put("/sheets/{userSheetId}/name", targetSheet.getUserSheetId())
                .contentType(MediaType.APPLICATION_JSON)
                .content(body))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.body.sheetName").value(newName))
                .andExpect(jsonPath("$.body.lastPracticeDate").exists())
                .andDo(print());
    }

    @Test
    @DisplayName("색상 변경 성공 테스트")
    void updateSheetColor_success() throws Exception {
        // given
        UserSheet targetSheet = userSheetRepository.findAll().get(0);
        String newColor = UUID.randomUUID().toString();
        SheetUpdateRequestDto requestDto = SheetUpdateRequestDto.builder()
                .email(targetSheet.getUserEmail())
                .color(newColor).build();
        List<SheetPractice> practiceList = sheetPracticeRepository.findAllByEmailAndSheetId(
                targetSheet.getUserEmail(),
                targetSheet.getUserSheetId(),
                PageRequest.of(0, 1));
        // when & then
        mockMvc.perform(put("/sheets/{userSheetId}/color", targetSheet.getUserSheetId())
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(new ObjectMapper().writeValueAsString(requestDto)))
                .andDo(print())
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.body.color").value(newColor))
                .andExpect(jsonPath("$.body.lastPracticeDate").exists());

    }

    @Test
    @DisplayName("악보 일괄 삭제 성공 테스트")
    void deleteSheet_success() throws Exception {
        // given
        String userEmail = TestDataGenerator.userEmails.get(0);
        List<Integer> userSheetIds = userSheetRepository.findAllByEmail(userEmail)
                .stream().map(UserSheet::getUserSheetId).toList();
        SheetListRequestDto requestDto = SheetListRequestDto.builder()
                .sheetIds(userSheetIds)
                .build();
        // when & then
        mockMvc.perform(delete("/sheets")
                .param("email", userEmail)
                .contentType(MediaType.APPLICATION_JSON)
                .content(new ObjectMapper().writeValueAsString(requestDto)))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.body").value(SuccessFlag.SUCCESS.getLabel()))
                .andDo(print());
        assert userSheetRepository.findAllByEmail(userEmail).isEmpty();
    }
}