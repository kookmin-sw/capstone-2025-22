package com.capstone.service;

import com.capstone.dto.ModelDto;
import com.capstone.dto.musicXml.MeasureInfo;
import com.capstone.dto.musicXml.NoteInfo;
import com.capstone.dto.musicXml.PitchInfo;
import com.capstone.dto.score.OnsetMatchResult;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

@Slf4j
@Component
public class PracticeResultResolver {

    public OnsetMatchResult matchOnset(ModelDto.OnsetResponseDto onsetResponse, MeasureInfo measureInfo){
        List<Double> userOnset = onsetResponse
                .getOnsets()
                .stream()
                .map(Double::parseDouble)
                .toList();
        List<Double> answerOnset = new ArrayList<>();
        double errorThreshold = 0.05;
        List<NoteInfo> noteInfoList = measureInfo.getNoteList();
        for(NoteInfo noteInfo : noteInfoList){
            answerOnset.add(noteInfo.getStartOnset());
        }
        return DTWMatcher.match(userOnset, answerOnset, errorThreshold);
    }

    public double calculateScore(List<Boolean> finalMatchingResult){
        int successCount = 0;
        for(Boolean result : finalMatchingResult){
            if(result){
                successCount++;
            }
        }
        return ((double) successCount / finalMatchingResult.size()) * 100;
    }

    public List<Boolean> calculateBeatMatchingResult(OnsetMatchResult onsetMatchResult){
        int[] matchedUserOnsetIndices = onsetMatchResult.getMatchedUserOnsetIndices();
        List<Double> answerOnsets = onsetMatchResult.getAnswerOnset();
        List<Boolean> beatMatchingResult = new ArrayList<>();
        for(int noteInfoIdx : matchedUserOnsetIndices) {
            if(noteInfoIdx < 0 || noteInfoIdx >= answerOnsets.size()) {
                beatMatchingResult.add(false);
            }else{
                beatMatchingResult.add(true);
            }
        }
        return beatMatchingResult;
    }

    public List<Boolean> calculateFinalMatchingResult(
            OnsetMatchResult onsetMatchResult,
            List<String[]> drumPredictList,
            MeasureInfo measureInfo){
        List<Boolean> finalMatchingResult = new ArrayList<>();
        List<NoteInfo> noteInfoList = measureInfo.getNoteList();
        int[] matchedUserOnsetIndices = onsetMatchResult.getMatchedUserOnsetIndices();
        for (int noteInfoIdx : matchedUserOnsetIndices) {
            if(noteInfoIdx >= 0 && noteInfoIdx < noteInfoList.size()){
                NoteInfo answerNoteInfo = noteInfoList.get(noteInfoIdx);
                String[] answerNotePrediction = answerNoteInfo.getPitchList()
                        .stream().map(PitchInfo::getInstrumentType).toArray(String[]::new);
                String[] userNotePrediction = drumPredictList.get(noteInfoIdx);
                Arrays.sort(answerNotePrediction);
                Arrays.sort(userNotePrediction);
                if (Arrays.equals(answerNotePrediction, userNotePrediction)) {
                    finalMatchingResult.add(true);
                }else{
                    finalMatchingResult.add(false);
                }
            }else{
                finalMatchingResult.add(false);
            }
        }
        return finalMatchingResult;
    }
}
