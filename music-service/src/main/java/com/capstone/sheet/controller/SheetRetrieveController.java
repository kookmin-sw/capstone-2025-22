package com.capstone.sheet.controller;

import com.capstone.response.ApiResponse;
import com.capstone.response.CustomResponseDto;
import com.capstone.sheet.dto.SheetListResponseDto;
import com.capstone.sheet.service.SheetRetrieveService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/sheets")
public class SheetRetrieveController {
    private final SheetRetrieveService sheetRetrieveService;
    public SheetRetrieveController(SheetRetrieveService sheetRetrieveService) {
        this.sheetRetrieveService = sheetRetrieveService;
    }
    @GetMapping("")
    public ResponseEntity<CustomResponseDto<SheetListResponseDto>> retrieveSheets(@RequestParam("email") String email) {
        return ApiResponse.success(new SheetListResponseDto(sheetRetrieveService.getSheetsByEmail(email)));
    }
}
