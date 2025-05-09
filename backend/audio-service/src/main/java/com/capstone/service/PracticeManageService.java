package com.capstone.service;

import com.capstone.dto.FinalMeasureResult;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.UUID;

@Service
@RequiredArgsConstructor
public class PracticeManageService {
    private final MeasureScoreManager measureScoreManager;

    public String createPracticeIdentifier(){
        return UUID.randomUUID().toString();
    }

    public boolean interruptPractice(String identifier){
        measureScoreManager.deleteAllMeasureScores(identifier);
        return true;
    }

    public boolean completePractice(String identifier){
        List<FinalMeasureResult> finalMeasureResults = measureScoreManager.getAllMeasureScores(identifier)
                .stream()
                .map(FinalMeasureResult::fromString)
                .toList();
        finalMeasureResults
                .stream()
                .mapToDouble(FinalMeasureResult::getScore)
                .average()
                .orElse(0.0);
        return true;
    }
}
