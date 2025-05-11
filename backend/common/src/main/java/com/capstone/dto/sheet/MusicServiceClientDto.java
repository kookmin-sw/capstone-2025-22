package com.capstone.dto.sheet;

import com.capstone.dto.score.FinalMeasureResult;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

public class MusicServiceClientDto {

    @Data
    @Builder
    @AllArgsConstructor
    @NoArgsConstructor
    public static class SheetPracticeCreateRequest {
        List<FinalMeasureResult> finalMeasures;
        String userEmail;
        int userSheetId;
        double score;

        public static SheetPracticeCreateRequest from(
                List<FinalMeasureResult> finalMeasureResults,
                int userSheetId,
                String userEmail) {
            double totalScore = 0;
            for(FinalMeasureResult finalMeasureResult : finalMeasureResults) {
                totalScore += finalMeasureResult.getScore();
            }
            totalScore/=finalMeasureResults.size();

            return SheetPracticeCreateRequest.builder()
                    .score(totalScore)
                    .finalMeasures(finalMeasureResults)
                    .userEmail(userEmail)
                    .userSheetId(userSheetId)
                    .build();
        }
    }
}
