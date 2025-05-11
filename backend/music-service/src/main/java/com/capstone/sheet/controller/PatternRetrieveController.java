package com.capstone.sheet.controller;

import com.capstone.response.ApiResponse;
import com.capstone.response.CustomResponseDto;
import com.capstone.sheet.dto.PatternResponseDto;
import com.capstone.sheet.service.PatternRetrieveService;
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
    public ResponseEntity<CustomResponseDto<List<PatternResponseDto>>> retrievePatterns(){
        return ApiResponse.success(patternRetrieveService.getAllPatterns());
    }

    @GetMapping("/{patternId}")
    public ResponseEntity<CustomResponseDto<PatternResponseDto>> retrievePatternById(@PathVariable("patternId") Long patternId){
        return ApiResponse.success(patternRetrieveService.getPatternById(patternId));
    }

    @GetMapping("/success")
    public ResponseEntity<CustomResponseDto<List<PatternResponseDto>>> retrieveSucceededPatterns(@RequestParam("email") String email){
        return ApiResponse.success(patternRetrieveService.getSucceededPatternsByEmail(email));
    }
}
