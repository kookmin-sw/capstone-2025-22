package com.capstone.sheet.controller;

import com.capstone.dto.musicXml.MeasureInfo;
import com.capstone.response.ApiResponse;
import com.capstone.response.CustomResponseDto;
import com.capstone.sheet.dto.PatternResponseDto;
import com.capstone.sheet.dto.PatternWavResponseDto;
import com.capstone.sheet.entity.Pattern;
import com.capstone.sheet.service.PatternRetrieveService;
import io.swagger.v3.oas.annotations.Operation;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/patterns")
@RequiredArgsConstructor
public class PatternRetrieveController {

    private final PatternRetrieveService patternRetrieveService;

    @GetMapping
    @Operation(summary = "모든 패턴 정보 조회")
    public ResponseEntity<CustomResponseDto<List<PatternResponseDto>>> retrievePatterns(){
        return ApiResponse.success(patternRetrieveService.getAllPatterns());
    }

    @GetMapping("/{patternId}")
    @Operation(summary = "특정 패턴 정보 조회")
    public ResponseEntity<CustomResponseDto<PatternResponseDto>> retrievePatternById(@PathVariable("patternId") Long patternId){
        return ApiResponse.success(patternRetrieveService.getPatternById(patternId));
    }

    @GetMapping("/success")
    @Operation(summary = "특정 사용자가 성공한 패턴 정보들 조회")
    public ResponseEntity<CustomResponseDto<List<PatternResponseDto>>> retrieveSucceededPatterns(@RequestParam("email") String email){
        return ApiResponse.success(patternRetrieveService.getSucceededPatternsByEmail(email));
    }

    @GetMapping("/details")
    @Operation(summary = "특정 사용자가 한번이라도 연습했던 패턴 정보들 조회")
    public ResponseEntity<CustomResponseDto<List<PatternResponseDto>>> retrievePatternsDetails(@RequestParam("email") String email){
        return ApiResponse.success(patternRetrieveService.getPatternsByEmail(email));
    }

    @GetMapping("/{patternId}/measures")
    @Operation(summary = "특정 패턴의 특정 마디 정보 조회")
    public ResponseEntity<CustomResponseDto<MeasureInfo>> retrievePatternMeasure(
            @PathVariable("patternId") Long patternId,
            @RequestParam("measureNumber") String measureNumber){
        return ApiResponse.success(patternRetrieveService.findMeasureInfo(patternId, measureNumber));
    }

    @GetMapping("/{patternId}/wavs")
    @Operation(summary = "특정 패턴의 샘플 wav 파일 조회")
    public ResponseEntity<CustomResponseDto<PatternWavResponseDto>> retrievePatternWav(@PathVariable("patternId") Long patternId){
        return ApiResponse.success(patternRetrieveService.findPatternWavInfo(patternId));
    }
}
