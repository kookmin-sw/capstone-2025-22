package com.capstone.practice.controller;

import com.capstone.dto.sheet.MusicServiceClientDto.PatternPracticeCreateRequest;
import com.capstone.enums.SuccessFlag;
import com.capstone.practice.dto.PatternPracticeResponseDto;
import com.capstone.practice.service.PatternPracticeManageService;
import com.capstone.response.ApiResponse;
import com.capstone.response.CustomResponseDto;
import io.swagger.v3.oas.annotations.Operation;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/patterns")
@RequiredArgsConstructor
public class PatternPracticeManageController {

    private final PatternPracticeManageService patternPracticeManageService;

    @PostMapping("/practices")
    @Operation(summary = "create pattern practice record")
    public ResponseEntity<CustomResponseDto<PatternPracticeResponseDto>> createPatternPractice(
            @RequestBody PatternPracticeCreateRequest createDto){
        return ApiResponse.success(patternPracticeManageService.savePractice(createDto));
    }

    @DeleteMapping("/practices")
    @Operation(summary = "delete pattern practice record with id")
    public ResponseEntity<CustomResponseDto<String>> deletePractice(@RequestParam("practiceId") int practiceId){
        patternPracticeManageService.deletePractice(practiceId);
        return ApiResponse.success(SuccessFlag.SUCCESS.getLabel());
    }
}
