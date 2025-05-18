package com.capstone.service;

import com.capstone.dto.ModelDto;
import com.capstone.dto.musicXml.MeasureInfo;
import com.capstone.dto.musicXml.NoteInfo;
import com.capstone.dto.musicXml.PitchInfo;
import com.capstone.dto.score.OnsetMatchResult;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

@Slf4j
@Component
public class PracticeResultResolver {

    @Value("${scoring.beat.threshold:0.2}")
    public Double errorThreshold = 0.2;

    public OnsetMatchResult matchOnset(ModelDto.OnsetResponseDto onsetResponse, MeasureInfo measureInfo, double weight){
        List<Double> userOnset = onsetResponse
                .getOnsets()
                .stream()
                .map(Double::parseDouble)
                .toList();
        List<Double> answerOnset = new ArrayList<>();
        List<NoteInfo> noteInfoList = measureInfo.getNoteList();
        for(NoteInfo noteInfo : noteInfoList){
            answerOnset.add(noteInfo.getStartOnset());
        }
        return DTWMatcher.match(userOnset, answerOnset, errorThreshold, weight);
    }

    public double calculateScore(
            List<Boolean> beatMatchingResult,
            List<Boolean> finalMatchingResult){
        int size = beatMatchingResult.size();
        double score = 0;
        double unitScore = (double) 100 / size;
        for(int i=0; i<size; i++){
            if(beatMatchingResult.get(i))
                score += (unitScore*0.7);
            if(finalMatchingResult.get(i))
                score += (unitScore*0.3);
        }
        return Math.ceil(score);
    }

    public List<Boolean> calculateBeatMatchingResult(OnsetMatchResult onsetMatchResult){
        int[] matchedUserOnsetIndices = onsetMatchResult.getMatchedUserOnsetIndices();
        List<Double> answerOnsets = onsetMatchResult.getAnswerOnset();
        List<Boolean> beatMatchingResult = new ArrayList<>();
        for(int i=0; i<answerOnsets.size(); i++){
            boolean flag = false;
            for (int matchedAnswerOnsetIdx : matchedUserOnsetIndices) {
                if (i == matchedAnswerOnsetIdx) {
                    flag = true;
                    break;
                }
            }
            beatMatchingResult.add(flag);
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
        for (int i=0; i<noteInfoList.size(); i++){
            boolean flag = false;
            for(int j=0; j<matchedUserOnsetIndices.length; j++){
                int matchedAnswerOnsetIdx = matchedUserOnsetIndices[j];
                if(matchedAnswerOnsetIdx < 0 || i != matchedAnswerOnsetIdx) continue;
                NoteInfo answerNoteInfo = noteInfoList.get(matchedAnswerOnsetIdx);
                String[] answerNotePrediction = answerNoteInfo.getPitchList()
                        .stream().map(PitchInfo::getInstrumentType).toArray(String[]::new);
                String[] userNotePrediction = drumPredictList.get(j);
                Arrays.sort(answerNotePrediction);
                Arrays.sort(userNotePrediction);
                if (Arrays.equals(answerNotePrediction, userNotePrediction)) {
                    flag = true;
                    break;
                }
            }
            finalMatchingResult.add(flag);
        }
        return finalMatchingResult;
    }
}
