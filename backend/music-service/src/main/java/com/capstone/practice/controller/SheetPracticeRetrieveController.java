package com.capstone.practice.controller;

import com.capstone.practice.dto.SheetPracticeDetailResponseDto;
import com.capstone.practice.dto.SheetPracticeRepresentResponse;
import com.capstone.practice.dto.SheetPracticeResponseDto;
import com.capstone.practice.service.SheetPracticeRetrieveService;
import com.capstone.response.ApiResponse;
import com.capstone.response.CustomResponseDto;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/sheets")
public class SheetPracticeRetrieveController {
    private final SheetPracticeRetrieveService sheetPracticeRetrieveService;
    public SheetPracticeRetrieveController(SheetPracticeRetrieveService sheetPracticeRetrieveService) {
        this.sheetPracticeRetrieveService = sheetPracticeRetrieveService;
    }
    /**
     * 특정 사용자의 특정 악보 연습 기록 전체 조회 (페지이네이션 적용)
     * @param userSheetId 악보 id
     * @param pageSize 조회할 연습 기록 최대 수
     * @param pageNumber pageSize*pageNumber 번째 연습기록부터 조회
     * @param email 사용자 이메일
     * @return 악보 연습 목록 반환
    * */
    @GetMapping("/{userSheetId}/practices")
    private ResponseEntity<CustomResponseDto<List<SheetPracticeResponseDto>>> getPractices(
            @PathVariable("userSheetId") int userSheetId,
            @RequestParam("pageSize") int pageSize,
            @RequestParam("pageNumber") int pageNumber,
            @RequestParam("email") String email) {
        List<SheetPracticeResponseDto> res = sheetPracticeRetrieveService.getSheetPracticeRecords(email, pageNumber, pageSize, userSheetId);
        return ApiResponse.success(res);
    }
    /**
     * 특정 악보 연습 기록 상세 조회
     * @param practiceId 연습 기록 id
     * @return 악보 연습 상세 정보 반환
    * */
    @GetMapping("/practices/{practiceId}")
    private ResponseEntity<CustomResponseDto<SheetPracticeDetailResponseDto>> getDetailPractice(@PathVariable("practiceId") int practiceId) {
        SheetPracticeDetailResponseDto res = sheetPracticeRetrieveService.getDetailSheetPracticeRecord(practiceId);
        return ApiResponse.success(res);
    }
    /**
     * 특정 악보의 대표 연습 정보 조회
     * @param userSheetId 사용자 악보 id
     * @param email 사용자 이메일
     * @return 악보 대표 연습 정보 반환
    * */
    @GetMapping("/{userSheetId}/practices/representative")
    private ResponseEntity<CustomResponseDto<SheetPracticeRepresentResponse>> getRepresentativePractice(
            @PathVariable("userSheetId") int userSheetId,
            @RequestParam("email") String email) {
        SheetPracticeRepresentResponse res = sheetPracticeRetrieveService.getRepresentSheetPractice(email, userSheetId);
        return ApiResponse.success(res);
    }
    /**
     * 특정 사용자의 대표 연습 정보 목록 조회
     * @param email 사용자 이메일
     * @return 악보 대표 연습 정보 목록
    * */
    @GetMapping("/practices/representative")
    public ResponseEntity<CustomResponseDto<List<SheetPracticeRepresentResponse>>> getRepresentativePractices(@RequestParam("email") String email){
        List<SheetPracticeRepresentResponse> res = sheetPracticeRetrieveService.getRepresentSheetPractices(email);
        return ApiResponse.success(res);
    }
}
