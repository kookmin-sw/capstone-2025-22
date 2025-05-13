package com.capstone.practice.controller;

import com.capstone.practice.dto.PatternPracticeResponseDto;
import com.capstone.practice.service.PatternPracticeRetrieveService;
import com.capstone.response.ApiResponse;
import com.capstone.response.CustomResponseDto;
import io.swagger.v3.oas.annotations.Operation;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/patterns")
@RequiredArgsConstructor
public class PatternPracticeRetrieveController {

    private final PatternPracticeRetrieveService patternPracticeRetrieveService;

    @GetMapping("/{patternId}/practices")
    @Operation(summary = "특정 사용자의 특정 패턴 및 필인 연습 이력 조회")
    public ResponseEntity<CustomResponseDto<List<PatternPracticeResponseDto>>> retrievePracticeRecordsByPattern(
            @PathVariable("patternId") Long patternId,
            @RequestParam("email") String userEmail){
        return ApiResponse.success(patternPracticeRetrieveService.getUserPracticeRecordsByPattern(userEmail, patternId));
    }
}
