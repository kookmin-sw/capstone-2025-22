package com.capstone.dto;

import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.extern.slf4j.Slf4j;

import java.util.List;

@Slf4j
@Data
@Builder
@AllArgsConstructor
@NoArgsConstructor
public class FinalMeasureResult {
    Double score;
    String measureNumber;
    OnsetMatchResult onsetMatchResult;
    List<String[]> userDrumPredictList;
    List<String[]> answerDrumPredictList;

    public static FinalMeasureResult fromString(String measureScoreString){
        try {
            return new ObjectMapper().readValue(measureScoreString, FinalMeasureResult.class);
        }catch (Exception e){
            log.error(e.getMessage(), e);
            throw new RuntimeException("failed to parse measure score string : " + measureScoreString);
        }
    }

    public String toString(){
        try {
            return new ObjectMapper().writeValueAsString(this);
        }catch (Exception e){
            log.error(e.getMessage(), e);
            throw new RuntimeException("failed to convert MeasureScore to String : " + this);
        }
    }
}
