package com.capstone.sheet.controller;

import com.capstone.data.TestDataGenerator;
import com.capstone.response.CustomResponseDto;
import com.capstone.sheet.dto.SheetDetailResponseDto;
import com.capstone.sheet.dto.SheetListResponseDto;
import com.capstone.sheet.dto.SheetResponseDto;
import com.capstone.sheet.entity.UserSheet;
import com.capstone.sheet.repository.UserSheetRepository;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.http.MediaType;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.test.web.servlet.MockMvc;

import java.time.LocalDateTime;
import java.util.Map;

import static org.hamcrest.Matchers.hasSize;
import static org.junit.jupiter.api.Assertions.*;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.result.MockMvcResultHandlers.print;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.content;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;

@SpringBootTest
@ActiveProfiles("test")
@AutoConfigureMockMvc
class SheetRetrieveControllerTest {
    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private TestDataGenerator testDataGenerator;

    @Autowired
    private UserSheetRepository userSheetRepository;

    @BeforeEach
    void setUp() {
        testDataGenerator.generateTestData();
    }

    @AfterEach
    void tearDown() {
        testDataGenerator.deleteAllTestData();
    }

    @Test
    void retrieveSheets_success() throws Exception {
        // given
        String email = TestDataGenerator.userEmails.get(0);
        SheetListResponseDto res = SheetListResponseDto.builder()
                .sheets(userSheetRepository
                        .findAllByEmail(email)
                        .stream()
                        .map(SheetResponseDto::from).toList())
                .build();
        // when & then
        mockMvc.perform(get("/sheets")
                .param("email", email)
                .contentType(MediaType.APPLICATION_JSON))
                .andExpect(jsonPath("$.body").exists())
                .andExpect(jsonPath("$.body.sheets").isArray())
                .andExpect(jsonPath("$.body.sheets", hasSize(res.getSheets().size())))
                .andDo(print());
    }

    @Test
    void retrieveSheetDetails_success() throws Exception {
        // given
        UserSheet userSheet = userSheetRepository.findAll().get(0);
        SheetDetailResponseDto res = SheetDetailResponseDto.from(userSheet, LocalDateTime.now());
        // when & then
        mockMvc.perform(get("/sheets/{userSheetId}", userSheet.getUserSheetId())
                        .contentType(MediaType.APPLICATION_JSON))
                .andExpect(jsonPath("$.body").exists())
                .andExpect(jsonPath("$.body.sheetName").value(res.getSheetName()))
                .andExpect(jsonPath("$.body.userSheetId").value(res.getUserSheetId()))
                .andDo(print());
    }
}