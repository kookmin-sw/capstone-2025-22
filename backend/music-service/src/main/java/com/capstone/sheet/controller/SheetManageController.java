package com.capstone.sheet.controller;

import com.capstone.enums.SuccessFlag;
import com.capstone.exception.InvalidRequestException;
import com.capstone.response.ApiResponse;
import com.capstone.response.CustomResponseDto;
import com.capstone.sheet.dto.SheetCreateRequestDto;
import com.capstone.sheet.dto.SheetListRequestDto;
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
    /**
     * 악보와 사용자 악보 생성
     * @param requestDto 악보 및 사용자별 악보 정보
     * @return SheetResponseDto
    * */
    @PostMapping("")
    public ResponseEntity<CustomResponseDto<SheetResponseDto>> createSheet(
            @RequestBody SheetCreateRequestDto requestDto) {
        return ApiResponse.success(sheetManageService.createSheetAndUserSheet(requestDto));
    }

    /**
     * 악보 이름 수정
     * @param userSheetId 사용자 악보 id
     * @param requestDto 사용자 이메일, 새로운 악보 이름
     * @return SheetResponseDto
    * */
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
    /**
     * 악보 색상 수정
     * @param userSheetId 사용자 악보 id
     * @param requestDto 사용자 이메일, 새로운 색상 정보
     * @return SheetResponseDto
     * */
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
    /**
     * 악보 일괄 삭제
     * @param email 사용자 이메일
     * @param requestDto 삭제할 악보 id 목록
     * @return valid | invalid
     * */
    @DeleteMapping("")
    public ResponseEntity<CustomResponseDto<String>> deleteSheet(
            @RequestParam("email") String email,
            @RequestBody SheetListRequestDto requestDto) {
        sheetManageService.deleteSheetByIdList(email, requestDto.sheetIds);
        return ApiResponse.success(SuccessFlag.SUCCESS.getLabel());
    }
}
