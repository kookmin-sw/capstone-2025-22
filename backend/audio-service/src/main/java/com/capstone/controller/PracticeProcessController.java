package com.capstone.controller;

import com.capstone.enums.SuccessFlag;
import com.capstone.response.ApiResponse;
import com.capstone.response.CustomResponseDto;
import com.capstone.service.PracticeManageService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/audio")
@RequiredArgsConstructor
public class PracticeProcessController {
    private final PracticeManageService practiceManageService;

    @PostMapping("/practice")
    public ResponseEntity<CustomResponseDto<String>> startPracticeProcess(){
        return ApiResponse.success(practiceManageService.createPracticeIdentifier());
    }

    @PostMapping("/practice/interruption")
    public ResponseEntity<CustomResponseDto<String>> pausePractice(@RequestParam String identifier){
        return practiceManageService.interruptPractice(identifier) ?
                ApiResponse.success(SuccessFlag.SUCCESS.getLabel()) :
                ApiResponse.success(SuccessFlag.FAILURE.getLabel());
    }

    @PostMapping("/practice/completion")
    public ResponseEntity<CustomResponseDto<String>> completePractice(@RequestParam String identifier){
        return practiceManageService.completePractice(identifier) ?
                ApiResponse.success(SuccessFlag.SUCCESS.getLabel()) :
                ApiResponse.success(SuccessFlag.FAILURE.getLabel());
    }
}
