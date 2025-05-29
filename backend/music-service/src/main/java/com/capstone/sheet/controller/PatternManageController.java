package com.capstone.sheet.controller;

import com.capstone.enums.SuccessFlag;
import com.capstone.response.ApiResponse;
import com.capstone.response.CustomResponseDto;
import com.capstone.sheet.dto.PatternCreateDto;
import com.capstone.sheet.dto.PatternResponseDto;
import com.capstone.sheet.dto.PatternUpdateDto;
import com.capstone.sheet.event.PatternCreateEvent;
import com.capstone.sheet.event.PatternUpdateEvent;
import com.capstone.sheet.service.PatternManageService;
import io.swagger.v3.oas.annotations.Operation;
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
    @Operation(summary = "create pattern data")
    public ResponseEntity<CustomResponseDto<String>> createPattern(
            @RequestPart PatternCreateDto patternCreateDto,
            @RequestPart MultipartFile sheetFile,
            @RequestPart MultipartFile patternWav) throws Exception{
        eventPublisher.publishEvent(new PatternCreateEvent(patternCreateDto, sheetFile.getBytes(), patternWav.getBytes()));
        return ApiResponse.success(SuccessFlag.SUCCESS.getLabel());
    }

    @PutMapping(value = "/{patternId}", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    @Operation(summary = "update pattern info by id")
    public ResponseEntity<CustomResponseDto<String>> updatePattern(
            @PathVariable("patternId") Long patternId,
            @RequestPart PatternUpdateDto patternUpdateDto,
            @RequestPart(required = false) MultipartFile sheetFile,
            @RequestPart(required = false) MultipartFile patternWav) throws Exception{
        byte[] sheetByte = sheetFile == null ? null : sheetFile.getBytes();
        byte[] patternWavByte = patternWav == null ? null : patternWav.getBytes();
        eventPublisher.publishEvent(new PatternUpdateEvent(patternId, patternUpdateDto, sheetByte, patternWavByte));
        return ApiResponse.success(SuccessFlag.SUCCESS.getLabel());
    }

    @DeleteMapping("/{patternId}")
    @Operation(summary = "delete pattern by pattern id")
    public ResponseEntity<CustomResponseDto<String>> deletePattern(@PathVariable("patternId") Long patternId){
        patternManageService.deletePattern(patternId);
        return ApiResponse.success(SuccessFlag.SUCCESS.getLabel());
    }
}
