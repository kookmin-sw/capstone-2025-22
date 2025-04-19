package com.capstone.sheet.controller;

import com.capstone.response.ApiResponse;
import com.capstone.response.CustomResponseDto;
import com.capstone.sheet.dto.SheetDetailResponseDto;
import com.capstone.sheet.dto.SheetListResponseDto;
import com.capstone.sheet.service.SheetRetrieveService;
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
    public ResponseEntity<CustomResponseDto<SheetListResponseDto>> retrieveSheets(@RequestParam("email") String email) {
        return ApiResponse.success(new SheetListResponseDto(sheetRetrieveService.getSheetsByEmail(email)));
    }
    /**
     * 특정 악보 정보 상세 조회
     * @param userSheetId 악보 id
    * */
    @GetMapping("/{userSheetId}")
    public ResponseEntity<CustomResponseDto<SheetDetailResponseDto>> retrieveSheetDetails(@PathVariable("userSheetId") int userSheetId) {
        return ApiResponse.success(sheetRetrieveService.getSheetById(userSheetId));
    }
}
