package com.capstone.sheet.controller;

import com.capstone.practice.entity.SheetPractice;
import com.capstone.practice.repository.SheetPracticeRepository;
import com.capstone.response.CustomResponseDto;
import com.capstone.sheet.dto.SheetResponseDto;
import com.capstone.sheet.dto.SheetUpdateRequestDto;
import com.capstone.sheet.entity.Sheet;
import com.capstone.sheet.entity.UserSheet;
import com.capstone.sheet.repository.SheetRepository;
import com.capstone.sheet.repository.UserSheetRepository;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.data.domain.PageRequest;
import org.springframework.http.MediaType;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.test.web.servlet.MockMvc;

import java.time.LocalDateTime;
import java.util.Arrays;
import java.util.List;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.*;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.put;
import static org.springframework.test.web.servlet.result.MockMvcResultHandlers.print;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

@SpringBootTest
@ActiveProfiles("test")
@AutoConfigureMockMvc
class SheetManageControllerTest {
    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private UserSheetRepository userSheetRepository;

    @Autowired
    private SheetRepository sheetRepository;

    @Autowired
    private SheetPracticeRepository sheetPracticeRepository;

    List<String> userEmails = Arrays.asList("test@gmail.com", "test2@gmail.com");

    @BeforeEach
    void setUp() {
        Sheet sheet = sheetRepository.save(
                Sheet.builder()
                        .sheetInfo("sheetInfo")
                        .build()
        );
        for (String userEmail : userEmails) {
            for(int i=0; i<5; i++) {
                UserSheet userSheet = userSheetRepository.save(
                        UserSheet.builder()
                                .sheetName("init")
                                .color("init")
                                .userEmail(userEmail)
                                .sheet(sheet)
                                .build()
                );
                sheetPracticeRepository.save(
                        SheetPractice.builder()
                                .userSheet(userSheet)
                                .score(80)
                                .practiceInfo("practiceInfo")
                                .userEmail(userEmail)
                                .createdDate(LocalDateTime.now())
                                .build()
                );
            }
        }
    }

    @AfterEach
    void tearDown() {
        sheetPracticeRepository.deleteAll();
        userSheetRepository.deleteAll();
        sheetRepository.deleteAll();
    }

    @Test
    @DisplayName("이름 변경 성공 테스트")
    void updateSheetName_success() throws Exception {
        // given : set request dto
        UserSheet targetSheet = userSheetRepository.findAll().get(0);
        String newName = UUID.randomUUID().toString();
        SheetUpdateRequestDto requestDto = SheetUpdateRequestDto.builder()
                .email(targetSheet.getUserEmail())
                .name(newName).build();
        List<SheetPractice> practiceList = sheetPracticeRepository.findAllByEmailAndSheetId(
                targetSheet.getUserEmail(),
                targetSheet.getUserSheetId(),
                PageRequest.of(0, 1));
        String lastPractice = practiceList.isEmpty() ? null : practiceList.get(0).getCreatedDate().toString();
        // when & then
        mockMvc.perform(put("/sheets/{userSheetId}/name", targetSheet.getUserSheetId())
                .contentType(MediaType.APPLICATION_JSON)
                .content(new ObjectMapper().writeValueAsString(requestDto)))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.body.sheetName").value(newName))
                .andExpect(jsonPath("$.body.lastPracticeDate").value(lastPractice))
                .andDo(print());
    }

    @Test
    @DisplayName("색상 변경 성공 테스트")
    void updateSheetColor_success() throws Exception {
        // given : set request dto
        UserSheet targetSheet = userSheetRepository.findAll().get(0);
        String newColor = UUID.randomUUID().toString();
        SheetUpdateRequestDto requestDto = SheetUpdateRequestDto.builder()
                .email(targetSheet.getUserEmail())
                .color(newColor).build();
        List<SheetPractice> practiceList = sheetPracticeRepository.findAllByEmailAndSheetId(
                targetSheet.getUserEmail(),
                targetSheet.getUserSheetId(),
                PageRequest.of(0, 1));
        String lastPractice = practiceList.isEmpty() ? null : practiceList.get(0).getCreatedDate().toString();
        // when & then
        mockMvc.perform(put("/sheets/{userSheetId}/color", targetSheet.getUserSheetId())
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(new ObjectMapper().writeValueAsString(requestDto)))
                .andDo(print())
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.body.color").value(newColor))
                .andExpect(jsonPath("$.body.lastPracticeDate").value(lastPractice));
    }
}