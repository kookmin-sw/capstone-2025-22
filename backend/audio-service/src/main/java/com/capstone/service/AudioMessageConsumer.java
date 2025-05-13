package com.capstone.service;

import com.capstone.client.AudioModelClient;
import com.capstone.client.MusicClientService;
import com.capstone.dto.AudioMessageDto;
import com.capstone.dto.PatternMessageDto;
import com.capstone.dto.score.FinalMeasureResult;
import com.capstone.dto.ModelDto.*;
import com.capstone.dto.score.OnsetMatchResult;
import com.capstone.dto.musicXml.MeasureInfo;
import lombok.Builder;
import lombok.Data;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.messaging.handler.annotation.Payload;
import org.springframework.messaging.simp.SimpMessagingTemplate;
import org.springframework.stereotype.Service;
import reactor.core.publisher.Mono;

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
    private final PracticeResultResolver practiceResultResolver;

    @Data
    @Builder
    private static class OnsetMeasureData{
        OnsetResponseDto onsetResponse;
        DrumPredictResponse drumPredictResponse;
        OnsetMatchResult onsetMatchResult;
        MeasureInfo measureInfo;
    }

    private OnsetMeasureData getOnsetResultAndSendToUser(OnsetMeasureData onsetMeasureData, String measureNumber, String email){
        OnsetResponseDto onsetResponse = onsetMeasureData.getOnsetResponse();
        MeasureInfo measureInfo = onsetMeasureData.getMeasureInfo();
        OnsetMatchResult matchResult = practiceResultResolver.matchOnset(onsetResponse, measureInfo);
        matchResult.setMeasureNumber(measureNumber);
        onsetMeasureData.setOnsetMatchResult(matchResult);
        messagingTemplate.convertAndSend("/topic/onset/" + email, matchResult);
        return onsetMeasureData;
    }

    private Mono<OnsetMeasureData> getDrumPredictionList(OnsetMeasureData onsetMeasureData, String base64Audio){
        OnsetResponseDto onsetResponse = onsetMeasureData.getOnsetResponse();
        DrumPredictRequest drumPredictRequest = DrumPredictRequest.builder()
                .audio_base64(base64Audio)
                .onsets(onsetResponse.getOnsets()).build();
        return audioModelClient.getDrumPredictions(drumPredictRequest)
                .map(drumPredictResponse -> {
                    onsetMeasureData.setDrumPredictResponse(drumPredictResponse);
                    return onsetMeasureData;
                });
    }

    private FinalMeasureResult getFinalMeasureResult(OnsetMeasureData onsetMeasureData, String measureNumber){
        OnsetMatchResult onsetMatchResult = onsetMeasureData.getOnsetMatchResult();
        List<String[]> predictList = onsetMeasureData.getDrumPredictResponse().getPredictions();
        MeasureInfo measureInfo = onsetMeasureData.getMeasureInfo();
        List<Boolean> beatScoringResults = practiceResultResolver.calculateBeatMatchingResult(onsetMatchResult);
        List<Boolean> finalScoringResults = practiceResultResolver.calculateFinalMatchingResult(onsetMatchResult, predictList, measureInfo);
        double score = practiceResultResolver.calculateScore(finalScoringResults);
        return FinalMeasureResult.builder()
                .measureNumber(measureNumber)
                .beatScoringResults(beatScoringResults)
                .finalScoringResults(finalScoringResults)
                .score(score).build();
    }

    @KafkaListener(topics = "audio", groupId = "${spring.kafka.consumer.group-id}")
    public void sendAudioConversionResult(@Payload final AudioMessageDto audioMessageDto) {
        OnsetRequestDto requestDto = OnsetRequestDto.fromMessageDto(audioMessageDto);
        audioModelClient.getOnsetFromWav(requestDto)
                .flatMap(onset -> musicClientService.getMeasureInfo(audioMessageDto.getUserSheetId(), audioMessageDto.getMeasureNumber())
                        .map(measureInfo -> OnsetMeasureData.builder()
                                .onsetResponse(onset)
                                .measureInfo(measureInfo)))
                .map(onsetMeasureDataBuilder -> { // get the onset match result and send it to the client
                    return getOnsetResultAndSendToUser(onsetMeasureDataBuilder.build(), audioMessageDto.getMeasureNumber(), audioMessageDto.getEmail());
                })
                .flatMap(onsetMeasureDataBuilder -> { // get the drum prediction list from the AudioModelClient
                    return getDrumPredictionList(onsetMeasureDataBuilder, audioMessageDto.getMessage());
                })
                .map(onsetMeasureDataBuilder -> { // get the final measure result and save it to the redis
                    FinalMeasureResult finalMeasureResult = getFinalMeasureResult(onsetMeasureDataBuilder, audioMessageDto.getMeasureNumber());
                    measureScoreManager.saveMeasureScore(audioMessageDto.getIdentifier(), audioMessageDto.getMeasureNumber(), finalMeasureResult);
                    return finalMeasureResult;
                })
                .filter(res -> audioMessageDto.isEndOfMeasure())
                .flatMap(finalMeasureResult -> { // save the final measure result list to the database
                    List<FinalMeasureResult> finalMeasureResultList = measureScoreManager.getAllMeasureScores(audioMessageDto.getIdentifier())
                            .stream().map(FinalMeasureResult::fromString).toList();
                    SheetPracticeCreateRequest sheetPracticeCreateRequest = SheetPracticeCreateRequest.from(finalMeasureResultList, audioMessageDto.getUserSheetId(), audioMessageDto.getEmail());
                    return musicClientService.saveMeasureScoreInfo(sheetPracticeCreateRequest).flatMap(saveRes -> {
                        if(!saveRes) {
                            log.error("[sheet practice] failed to save data : retry again");
                            return musicClientService.saveMeasureScoreInfo(sheetPracticeCreateRequest);
                        }
                        return Mono.empty();
                    });
                }).subscribe();
    }

    @KafkaListener(topics = "pattern")
    public void sendPatternResultAndSavePatternPractice(@Payload final PatternMessageDto patternMessageDto){
        OnsetRequestDto requestDto = OnsetRequestDto.fromPatternMessage(patternMessageDto);
        audioModelClient.getOnsetFromWav(requestDto)
                .flatMap(onset -> musicClientService.getPatternMeasureInfo(patternMessageDto.getPatternId(), patternMessageDto.getMeasureNumber())
                        .map(measureInfo -> OnsetMeasureData.builder()
                                .onsetResponse(onset)
                                .measureInfo(measureInfo)))
                .map(onsetMeasureDataBuilder -> { // get the onset match result and send it to the client
                    return getOnsetResultAndSendToUser(onsetMeasureDataBuilder.build(), patternMessageDto.getMeasureNumber(), patternMessageDto.getEmail());
                })
                .flatMap(onsetMeasureDataBuilder -> { // get the drum prediction list from the AudioModelClient
                    return getDrumPredictionList(onsetMeasureDataBuilder, patternMessageDto.getAudioBase64());
                })
                .map(onsetMeasureDataBuilder -> { // get the final measure result and save it to the redis
                    FinalMeasureResult finalMeasureResult = getFinalMeasureResult(onsetMeasureDataBuilder, patternMessageDto.getMeasureNumber());
                    measureScoreManager.saveMeasureScore(patternMessageDto.getIdentifier(), patternMessageDto.getMeasureNumber(), finalMeasureResult);
                    return finalMeasureResult;
                })
                .filter(res -> patternMessageDto.isEndOfMeasure())
                .flatMap(finalMeasureResult -> { // save the final measure result list of pattern practice to the database
                    List<FinalMeasureResult> finalMeasureResultList = measureScoreManager.getAllMeasureScores(patternMessageDto.getIdentifier())
                            .stream().map(FinalMeasureResult::fromString).toList();
                    PatternPracticeCreateRequest createDto = PatternPracticeCreateRequest.from(finalMeasureResultList, patternMessageDto.getPatternId(), patternMessageDto.getEmail());
                    return musicClientService.savePatternScoreInfo(createDto).flatMap(saveRes -> {
                        if(!saveRes) {
                            log.error("[pattern practice] failed to save data : retry again");
                            return musicClientService.savePatternScoreInfo(createDto);
                        }
                        return Mono.empty();
                    });
                }).subscribe();
    }
}
