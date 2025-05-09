package com.capstone.service;

import com.capstone.client.AudioModelClient;
import com.capstone.client.MusicClientService;
import com.capstone.dto.AudioMessageDto;
import com.capstone.dto.score.FinalMeasureResult;
import com.capstone.dto.ModelDto.*;
import com.capstone.dto.score.OnsetMatchResult;
import com.capstone.dto.musicXml.MeasureInfo;
import com.capstone.dto.musicXml.NoteInfo;
import com.capstone.dto.musicXml.PitchInfo;
import lombok.Builder;
import lombok.Data;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.messaging.handler.annotation.Payload;
import org.springframework.messaging.simp.SimpMessagingTemplate;
import org.springframework.stereotype.Service;
import reactor.core.scheduler.Schedulers;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static com.capstone.dto.sheet.MusicServiceClientDto.*;

@Slf4j
@Service
@RequiredArgsConstructor
public class AudioMessageConsumer {
    private final SimpMessagingTemplate messagingTemplate;
    private final AudioModelClient audioModelClient;
    private final MusicClientService musicClientService;
    private final MeasureScoreManager measureScoreManager;

    @Data
    @Builder
    private static class OnsetMeasureData{
        OnsetResponseDto onsetResponse;
        OnsetMatchResult onsetMatchResult;
        MeasureInfo measureInfo;
    }

    public OnsetMatchResult matchOnset(OnsetResponseDto onsetResponse, MeasureInfo measureInfo){
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
        List<Double> answerOnsets = onsetMatchResult.getAnswerOnset();
        int[] matchedUserOnsetIndices = onsetMatchResult.getMatchedUserOnsetIndices();
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
            }
        }
        return finalMatchingResult;
    }

    @KafkaListener(topics = "audio", groupId = "${spring.kafka.consumer.group-id}")
    public void sendAudioConversionResult(@Payload final AudioMessageDto audioMessageDto) {
        OnsetRequestDto requestDto = OnsetRequestDto.fromMessageDto(audioMessageDto);
        audioModelClient.getOnsetFromWav(requestDto)
                .flatMap(onset -> musicClientService.getMeasureInfo(audioMessageDto.getUserSheetId(), audioMessageDto.getMeasureNumber())
                        .map(measureInfo -> OnsetMeasureData.builder()
                                .onsetResponse(onset)
                                .measureInfo(measureInfo)))
                .map(onsetMeasureDataBuilder -> {
                    OnsetResponseDto onsetResponse = onsetMeasureDataBuilder.build().getOnsetResponse();
                    MeasureInfo measureInfo = onsetMeasureDataBuilder.build().getMeasureInfo();
                    OnsetMatchResult matchResult = matchOnset(onsetResponse, measureInfo);
                    messagingTemplate.convertAndSend("/topic/onset/" + audioMessageDto.getEmail(), matchResult);
                    return onsetMeasureDataBuilder.onsetMatchResult(matchResult);
                })
                .publishOn(Schedulers.boundedElastic())
                .doOnSuccess(onsetMeasureDataBuilder ->{
                    OnsetMatchResult onsetMatchResult = onsetMeasureDataBuilder.build().getOnsetMatchResult();
                    MeasureInfo measureInfo = onsetMeasureDataBuilder.build().getMeasureInfo();
                    audioModelClient.getDrumPredictions(DrumPredictRequest.builder()
                                    .audio_base64(audioMessageDto.getMessage())
                                    .onsets(onsetMeasureDataBuilder.build().getOnsetResponse().getOnsets())
                                    .build())
                            .doOnSuccess(drumPredictResponse -> {
                                List<Boolean> beatScoringResults = calculateBeatMatchingResult(onsetMatchResult);
                                List<Boolean> finalScoringResults = calculateFinalMatchingResult(onsetMatchResult, drumPredictResponse.getPredictions(), measureInfo);
                                double score = calculateScore(finalScoringResults);
                                FinalMeasureResult finalMeasureResult = FinalMeasureResult.builder()
                                        .measureNumber(audioMessageDto.getMeasureNumber())
                                        .beatScoringResults(beatScoringResults)
                                        .finalScoringResults(finalScoringResults)
                                        .score(score).build();
                                measureScoreManager.saveMeasureScore(audioMessageDto.getIdentifier(), audioMessageDto.getMeasureNumber(), finalMeasureResult);
                            })
                            .doOnSuccess(res -> {
                                if(audioMessageDto.isLastMeasure()){
                                    List<FinalMeasureResult> finalMeasureResults = measureScoreManager
                                            .getAllMeasureScores(audioMessageDto.getIdentifier())
                                            .stream()
                                            .map(FinalMeasureResult::fromString)
                                            .toList();
                                    musicClientService.saveMeasureScoreInfo(SheetPracticeCreateRequest.from(finalMeasureResults, audioMessageDto.getUserSheetId(), audioMessageDto.getEmail()))
                                            .subscribeOn(Schedulers.boundedElastic())
                                            .doOnSuccess(isSaveSuccess -> {
                                                if(!isSaveSuccess) {
                                                    log.error("failed to save data : retry again");
                                                    musicClientService.saveMeasureScoreInfo(
                                                            SheetPracticeCreateRequest.from(finalMeasureResults, audioMessageDto.getUserSheetId(), audioMessageDto.getEmail())
                                                    ).block();
                                                }
                                            }).subscribe();
                                }
                            }).subscribe();
                }).subscribe();
    }
}
