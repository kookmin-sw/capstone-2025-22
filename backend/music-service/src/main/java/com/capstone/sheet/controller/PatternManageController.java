package com.capstone.sheet.controller;

import com.capstone.enums.SuccessFlag;
import com.capstone.response.ApiResponse;
import com.capstone.response.CustomResponseDto;
import com.capstone.sheet.dto.PatternCreateDto;
import com.capstone.sheet.dto.PatternResponseDto;
import com.capstone.sheet.event.PatternCreateEvent;
import com.capstone.sheet.service.PatternManageService;
import lombok.RequiredArgsConstructor;
import org.springframework.context.ApplicationEventPublisher;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

@RestController
@RequestMapping("/patterns")
@RequiredArgsConstructor
public class PatternManageController {

    private final PatternManageService patternManageService;

    private final ApplicationEventPublisher eventPublisher;

    @PostMapping(consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<CustomResponseDto<String>> createPattern(
            @RequestPart PatternCreateDto patternCreateDto,
            @RequestPart MultipartFile sheetFile) throws Exception{
        eventPublisher.publishEvent(new PatternCreateEvent(patternCreateDto, sheetFile.getBytes()));
        return ApiResponse.success(SuccessFlag.SUCCESS.getLabel());
    }

    @PutMapping("/{patternId}")
    public ResponseEntity<CustomResponseDto<PatternResponseDto>> updatePattern(@PathVariable("patternId") Long patternId, @RequestBody PatternCreateDto patternCreateDto){
        return ApiResponse.success(patternManageService.updatePattern(patternId, patternCreateDto));
    }

    @DeleteMapping("/{patternId}")
    public ResponseEntity<CustomResponseDto<String>> deletePattern(@PathVariable("patternId") Long patternId){
        patternManageService.deletePattern(patternId);
        return ApiResponse.success(SuccessFlag.SUCCESS.getLabel());
    }
}
