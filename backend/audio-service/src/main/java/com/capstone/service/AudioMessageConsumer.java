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
import reactor.core.publisher.Flux;
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

    private OnsetMeasureData getOnsetResultAndSendToUser(OnsetMeasureData onsetMeasureData, String measureNumber, String email, String identifier, double weight){
        OnsetResponseDto onsetResponse = onsetMeasureData.getOnsetResponse();
        MeasureInfo measureInfo = onsetMeasureData.getMeasureInfo();
        OnsetMatchResult matchResult = practiceResultResolver.matchOnset(onsetResponse, measureInfo, weight);
        matchResult.setMeasureNumber(measureNumber);
        onsetMeasureData.setOnsetMatchResult(matchResult);
        log.info("[onset match] {}", matchResult);
        String destination = String.format("/topic/onset/%s/%s", email, identifier);
        messagingTemplate.convertAndSend(destination, matchResult);
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
        double score = practiceResultResolver.calculateScore(beatScoringResults, finalScoringResults);
        return FinalMeasureResult.builder()
                .measureNumber(measureNumber)
                .beatScoringResults(beatScoringResults)
                .finalScoringResults(finalScoringResults)
                .score(score).build();
    }

    @KafkaListener(topics = "audio", groupId = "${spring.kafka.consumer.group-id}", containerFactory = "audioKafkaListenerFactory")
    public Mono<Void> sendAudioConversionResult(@Payload final AudioMessageDto audioMessageDto) {
        OnsetRequestDto requestDto = OnsetRequestDto.fromMessageDto(audioMessageDto);
        return audioModelClient.getOnsetFromWav(requestDto)
                .flatMap(onset -> musicClientService.getMeasureInfo(audioMessageDto.getUserSheetId(), audioMessageDto.getMeasureNumber())
                        .map(measureInfo -> OnsetMeasureData.builder()
                                .onsetResponse(onset)
                                .measureInfo(measureInfo)))
                .map(onsetMeasureDataBuilder -> { // get the onset match result and send it to the client
                    double weight = (double) 60 / audioMessageDto.getBpm();
                    return getOnsetResultAndSendToUser(onsetMeasureDataBuilder.build(), audioMessageDto.getMeasureNumber(), audioMessageDto.getEmail(), audioMessageDto.getIdentifier(), weight);
                })
                .flatMap(onsetMeasureDataBuilder -> { // get the drum prediction list from the AudioModelClient
                    return getDrumPredictionList(onsetMeasureDataBuilder, audioMessageDto.getMessage());
                })
                .flatMap(onsetMeasureDataBuilder -> { // get the final measure result and save it to the redis
                    FinalMeasureResult finalMeasureResult = getFinalMeasureResult(onsetMeasureDataBuilder, audioMessageDto.getMeasureNumber());
                    return measureScoreManager.saveMeasureScore(audioMessageDto.getIdentifier(), audioMessageDto.getMeasureNumber(), finalMeasureResult)
                            .thenReturn(finalMeasureResult);
                })
                .filter(res -> audioMessageDto.isEndOfMeasure())
                .flatMap(finalMeasureResult -> measureScoreManager.getAllMeasureScores(audioMessageDto.getIdentifier())
                        .flatMapMany(scoreStrings -> Flux.fromIterable(scoreStrings)
                                .map(FinalMeasureResult::fromString))
                        .collectList()
                        .flatMap(finalMeasureResultList -> {
                            SheetPracticeCreateRequest sheetPracticeCreateRequest = SheetPracticeCreateRequest.from(finalMeasureResultList, audioMessageDto.getUserSheetId(), audioMessageDto.getEmail());
                            return musicClientService.saveMeasureScoreInfo(sheetPracticeCreateRequest).flatMap(saveRes -> {
                                if(!saveRes) {
                                    log.error("[sheet practice] failed to save data : retry again");
                                    return musicClientService.saveMeasureScoreInfo(sheetPracticeCreateRequest);
                                }
                                return Mono.empty();
                            });
                        })).then();
    }

    @KafkaListener(topics = "pattern", containerFactory = "patternKafkaListenerFactory")
    public void sendPatternResultAndSavePatternPractice(@Payload final PatternMessageDto patternMessageDto){
        OnsetRequestDto requestDto = OnsetRequestDto.fromPatternMessage(patternMessageDto);
        audioModelClient.getOnsetFromWav(requestDto)
                .flatMap(onset -> musicClientService.getPatternMeasureInfo(patternMessageDto.getPatternId(), patternMessageDto.getMeasureNumber())
                        .map(measureInfo -> OnsetMeasureData.builder()
                                .onsetResponse(onset)
                                .measureInfo(measureInfo)))
                .map(onsetMeasureDataBuilder -> { // get the onset match result and send it to the client
                    double weight = (double) 60 / patternMessageDto.getBpm();
                    return getOnsetResultAndSendToUser(onsetMeasureDataBuilder.build(), patternMessageDto.getMeasureNumber(), patternMessageDto.getEmail(), patternMessageDto.getIdentifier(), weight);
                })
                .flatMap(onsetMeasureDataBuilder -> { // get the drum prediction list from the AudioModelClient
                    return getDrumPredictionList(onsetMeasureDataBuilder, patternMessageDto.getAudioBase64());
                })
                .flatMap(onsetMeasureDataBuilder -> { // get the final measure result and save it to the redis
                    FinalMeasureResult finalMeasureResult = getFinalMeasureResult(onsetMeasureDataBuilder, patternMessageDto.getMeasureNumber());
                    return measureScoreManager
                            .saveMeasureScore(patternMessageDto.getIdentifier(), patternMessageDto.getMeasureNumber(), finalMeasureResult)
                            .filter(saved -> saved)
                            .thenReturn(finalMeasureResult);
                })
                .filter(res -> patternMessageDto.isEndOfMeasure())
                .flatMap(finalMeasureResult -> { // save the final measure result list of pattern practice to the database
                    return measureScoreManager.getAllMeasureScores(patternMessageDto.getIdentifier())
                            .flatMapMany(scoreStrings -> Flux.fromIterable(scoreStrings)
                                    .map(FinalMeasureResult::fromString))
                            .collectList()
                            .flatMap(finalMeasureResultList -> {
                                PatternPracticeCreateRequest createDto = PatternPracticeCreateRequest.from(finalMeasureResultList, patternMessageDto.getPatternId(), patternMessageDto.getEmail());
                                return musicClientService.savePatternScoreInfo(createDto).flatMap(saveRes -> {
                                    if(!saveRes) {
                                        log.error("[pattern practice] failed to save data : retry again");
                                        return musicClientService.savePatternScoreInfo(createDto);
                                    }
                                    return Mono.empty();
                                });
                            });
                }).subscribe(
                        unused -> {},
                        error -> log.error("[pattern practice] Error during processing", error)
                );
    }
}
