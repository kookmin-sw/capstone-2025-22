package com.capstone.service;

import com.capstone.client.AudioModelClient;
import com.capstone.client.MusicClientService;
import com.capstone.dto.AudioMessageDto;
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

    @KafkaListener(topics = "audio", groupId = "${spring.kafka.consumer.group-id}")
    public void sendAudioConversionResult(@Payload final AudioMessageDto audioMessageDto) {
        OnsetRequestDto requestDto = OnsetRequestDto.fromMessageDto(audioMessageDto);
        audioModelClient.getOnsetFromWav(requestDto)
                .flatMap(onset -> musicClientService.getMeasureInfo(audioMessageDto.getUserSheetId(), audioMessageDto.getMeasureNumber())
                        .map(measureInfo -> OnsetMeasureData.builder()
                                .onsetResponse(onset)
                                .measureInfo(measureInfo)))
                .map(onsetMeasureDataBuilder -> { // get the onset match result and send it to the client
                    OnsetResponseDto onsetResponse = onsetMeasureDataBuilder.build().getOnsetResponse();
                    MeasureInfo measureInfo = onsetMeasureDataBuilder.build().getMeasureInfo();
                    OnsetMatchResult matchResult = practiceResultResolver.matchOnset(onsetResponse, measureInfo);
                    matchResult.setMeasureNumber(audioMessageDto.getMeasureNumber());
                    messagingTemplate.convertAndSend("/topic/onset/" + audioMessageDto.getEmail(), matchResult);
                    return onsetMeasureDataBuilder.onsetMatchResult(matchResult);
                })
                .flatMap(onsetMeasureDataBuilder -> { // get the drum prediction list from the AudioModelClient
                    OnsetResponseDto onsetResponse = onsetMeasureDataBuilder.build().getOnsetResponse();
                    DrumPredictRequest drumPredictRequest = DrumPredictRequest.builder()
                            .audio_base64(audioMessageDto.getMessage())
                            .onsets(onsetResponse.getOnsets()).build();
                    return audioModelClient.getDrumPredictions(drumPredictRequest)
                            .map(drumPredictResponse -> {
                                onsetMeasureDataBuilder.drumPredictResponse(drumPredictResponse);
                                return onsetMeasureDataBuilder;
                            });
                })
                .map(onsetMeasureDataBuilder -> { // get the final measure result and save it to the redis
                    OnsetMeasureData built = onsetMeasureDataBuilder.build();
                    OnsetMatchResult onsetMatchResult = built.getOnsetMatchResult();
                    List<String[]> predictList = built.getDrumPredictResponse().getPredictions();
                    MeasureInfo measureInfo = built.getMeasureInfo();
                    List<Boolean> beatScoringResults = practiceResultResolver.calculateBeatMatchingResult(onsetMatchResult);
                    List<Boolean> finalScoringResults = practiceResultResolver.calculateFinalMatchingResult(onsetMatchResult, predictList, measureInfo);
                    double score = practiceResultResolver.calculateScore(finalScoringResults);
                    FinalMeasureResult finalMeasureResult = FinalMeasureResult.builder()
                            .measureNumber(audioMessageDto.getMeasureNumber())
                            .beatScoringResults(beatScoringResults)
                            .finalScoringResults(finalScoringResults)
                            .score(score).build();
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
                            log.error("failed to save data : retry again");
                            return musicClientService.saveMeasureScoreInfo(sheetPracticeCreateRequest);
                        }
                        return Mono.empty();
                    });
                }).subscribe();
    }
}
