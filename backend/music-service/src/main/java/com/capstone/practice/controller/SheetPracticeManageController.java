package com.capstone.practice.controller;

import com.capstone.dto.sheet.MusicServiceClientDto;
import com.capstone.enums.SuccessFlag;
import com.capstone.practice.service.SheetPracticeManageService;
import com.capstone.response.ApiResponse;
import com.capstone.response.CustomResponseDto;
import io.swagger.v3.oas.annotations.Operation;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/sheets")
@RequiredArgsConstructor
public class SheetPracticeManageController {

    private final SheetPracticeManageService sheetPracticeManageService;

    @PostMapping("/{userSheetId}/practices")
    @Operation(summary = "create sheet practice record")
    public ResponseEntity<CustomResponseDto<String>> createPractice(@PathVariable("userSheetId") int userSheetId, @RequestBody MusicServiceClientDto.SheetPracticeCreateRequest sheetPracticeCreateRequest) {
        sheetPracticeManageService.saveSheetPractice(sheetPracticeCreateRequest);
        return ApiResponse.success(SuccessFlag.SUCCESS.getLabel());
    }

    @DeleteMapping("/practices/{sheetPracticeId}")
    @Operation(summary = "delete practice record by id")
    public ResponseEntity<CustomResponseDto<String>> deletePractice(@PathVariable("sheetPracticeId") int sheetPracticeId) {
        sheetPracticeManageService.deleteSheetPractice(sheetPracticeId);
        return ApiResponse.success(SuccessFlag.SUCCESS.getLabel());
    }
}
