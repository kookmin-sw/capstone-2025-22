package com.capstone.sheet.controller;

import com.capstone.enums.SuccessFlag;
import com.capstone.exception.InvalidRequestException;
import com.capstone.response.ApiResponse;
import com.capstone.response.CustomResponseDto;
import com.capstone.sheet.dto.SheetCreateMeta;
import com.capstone.sheet.dto.SheetListRequestDto;
import com.capstone.sheet.dto.SheetResponseDto;
import com.capstone.sheet.dto.SheetUpdateRequestDto;
import com.capstone.sheet.service.SheetManageService;
import com.capstone.sheet.service.SheetUpdateService;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.media.Encoding;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

@RestController
@RequestMapping("/sheets")
public class SheetManageController {
    private final SheetUpdateService sheetUpdateService;
    private final SheetManageService sheetManageService;
    public SheetManageController(SheetUpdateService sheetUpdateService, SheetManageService sheetManageService) {
        this.sheetUpdateService = sheetUpdateService;
        this.sheetManageService = sheetManageService;
    }
    /**
     * 악보 생성
     * @param sheetCreateMeta 악보 생성을 위한 정보들
     * @param sheetFile 악보 파일 (pdf 혹은 이미지)
    * */
    @PostMapping(consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<CustomResponseDto<SheetResponseDto>> createSheet(
            @RequestPart("sheetCreateMeta") SheetCreateMeta sheetCreateMeta,
            @RequestPart("sheetFile") MultipartFile sheetFile){
        return ApiResponse.success(sheetManageService.saveSheetAndUserSheet(sheetCreateMeta, sheetFile));
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
        SheetResponseDto updatedUserSheet = sheetUpdateService.updateSheetName(
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
        SheetResponseDto updatedUserSheet = sheetUpdateService.updateSheetColor(
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
        sheetUpdateService.deleteSheetByIdList(email, requestDto.sheetIds);
        return ApiResponse.success(SuccessFlag.SUCCESS.getLabel());
    }
}
