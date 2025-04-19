package com.capstone.sheet.controller;

import com.capstone.data.TestDataGenerator;
import com.capstone.enums.SuccessFlag;
import com.capstone.practice.entity.SheetPractice;
import com.capstone.practice.repository.SheetPracticeRepository;
import com.capstone.sheet.dto.SheetListRequestDto;
import com.capstone.sheet.dto.SheetUpdateRequestDto;
import com.capstone.sheet.entity.UserSheet;
import com.capstone.sheet.repository.UserSheetRepository;
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

import java.util.List;
import java.util.UUID;
import java.util.stream.Collectors;

import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.delete;
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
    private SheetPracticeRepository sheetPracticeRepository;

    @Autowired
    TestDataGenerator testDataGenerator;

    @BeforeEach
    void setUp() {
        testDataGenerator.generateTestData();
    }

    @AfterEach
    void tearDown() {
        testDataGenerator.deleteAllTestData();
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