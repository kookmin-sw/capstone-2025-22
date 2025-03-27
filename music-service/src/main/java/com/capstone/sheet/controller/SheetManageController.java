package com.capstone.sheet.controller;

import com.capstone.enums.SuccessFlag;
import com.capstone.exception.InvalidRequestException;
import com.capstone.response.ApiResponse;
import com.capstone.response.CustomResponseDto;
import com.capstone.sheet.dto.SheetResponseDto;
import com.capstone.sheet.dto.SheetUpdateRequestDto;
import com.capstone.sheet.service.SheetManageService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/sheets")
public class SheetManageController {
    private final SheetManageService sheetManageService;
    public SheetManageController(SheetManageService sheetManageService) {
        this.sheetManageService = sheetManageService;
    }
    @GetMapping("/health")
    public ResponseEntity<CustomResponseDto<String>> health() {
        return ApiResponse.success(SuccessFlag.SUCCESS.getLabel());
    }

    @PutMapping("/{userSheetId}/name")
    public ResponseEntity<CustomResponseDto<SheetResponseDto>> updateSheetName(
            @PathVariable("userSheetId") int userSheetId,
            @RequestBody SheetUpdateRequestDto requestDto) {
        if(requestDto.getName()==null || requestDto.getEmail()==null){
            throw new InvalidRequestException("name or email or userSheetId is required");
        }
        SheetResponseDto updatedUserSheet = sheetManageService.updateSheetName(
                requestDto.getEmail(),
                requestDto.getName(),
                userSheetId);
        return ApiResponse.success(updatedUserSheet);
    }
    @PutMapping("/{userSheetId}/color")
    public ResponseEntity<CustomResponseDto<SheetResponseDto>> updateSheetColor(
    @PathVariable("userSheetId") int userSheetId,
    @RequestBody SheetUpdateRequestDto requestDto) {
        if(requestDto.getColor()==null || requestDto.getEmail()==null){
            throw new InvalidRequestException("name or email or userSheetId is required");
        }
        SheetResponseDto updatedUserSheet = sheetManageService.updateSheetColor(
                requestDto.getEmail(),
                requestDto.getColor(),
                userSheetId);
        return ApiResponse.success(updatedUserSheet);
    }
}
