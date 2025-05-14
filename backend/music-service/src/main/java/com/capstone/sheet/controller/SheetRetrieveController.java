package com.capstone.sheet.controller;

import com.capstone.response.ApiResponse;
import com.capstone.response.CustomResponseDto;
import com.capstone.sheet.dto.SheetDetailResponseDto;
import com.capstone.sheet.dto.SheetListResponseDto;
import com.capstone.dto.musicXml.MeasureInfo;
import com.capstone.sheet.service.SheetRetrieveService;
import io.swagger.v3.oas.annotations.Operation;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/sheets")
public class SheetRetrieveController {
    private final SheetRetrieveService sheetRetrieveService;
    public SheetRetrieveController(SheetRetrieveService sheetRetrieveService) {
        this.sheetRetrieveService = sheetRetrieveService;
    }
    /**
     * 사용자의 악보 모두 조회
     * @param email 사용자 이메일
     * @return SheetResponseDto
    * */
    @GetMapping("")
    @Operation(summary = "retrieve all user sheets by email")
    public ResponseEntity<CustomResponseDto<SheetListResponseDto>> retrieveSheets(@RequestParam("email") String email) {
        return ApiResponse.success(new SheetListResponseDto(sheetRetrieveService.getSheetsByEmail(email)));
    }
    /**
     * 특정 악보 정보 상세 조회
     * @param userSheetId 악보 id
     * @return CustomResponseDto<SheetDetailResponseDto>
    * */
    @GetMapping("/{userSheetId}")
    @Operation(summary = "retrieve user sheet's detail info by id")
    public ResponseEntity<CustomResponseDto<SheetDetailResponseDto>> retrieveSheetDetails(@PathVariable("userSheetId") int userSheetId) {
        return ApiResponse.success(sheetRetrieveService.getSheetById(userSheetId));
    }
    /**
     * 특정 악보의 특정 마디 정보 조회
     * @param userSheetId user sheet id
     * @param measureNumber measure tag's attribute value (number)
     * @return CustomResponseDto<MeasureInfo>
    * */
    @GetMapping("/{userSheetId}/measures")
    @Operation(summary = "get sheet's measure info by user sheet and measure number")
    public ResponseEntity<CustomResponseDto<MeasureInfo>> retrieveSheetMeasure(
            @PathVariable("userSheetId") int userSheetId,
            @RequestParam("measureNumber") String measureNumber) {
        MeasureInfo measureInfo = sheetRetrieveService.findMeasureInfo(userSheetId, measureNumber);
        return ApiResponse.success(measureInfo);
    }
}
