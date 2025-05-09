package com.capstone.service;

import com.capstone.client.AudioModelClient;
import com.capstone.client.MusicClientService;
import com.capstone.dto.AudioMessageDto;
import com.capstone.dto.FinalMeasureResult;
import com.capstone.dto.ModelDto.*;
import com.capstone.dto.OnsetMatchResult;
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

    public double getScore(OnsetMatchResult onsetMatchResult, List<String[]> drumPredictList, MeasureInfo measureInfo){
        int correctNoteCount = 0;
        List<NoteInfo> noteInfoList = measureInfo.getNoteList();
        int[] matchedUserOnsetIndices = onsetMatchResult.getMatchedUserOnsetIndices();
        for (int noteInfoIdx : matchedUserOnsetIndices) {
            if (noteInfoIdx < 0 || noteInfoIdx >= noteInfoList.size()) continue;
            NoteInfo answerNoteInfo = noteInfoList.get(noteInfoIdx);
            String[] answerNotePrediction = answerNoteInfo.getPitchList()
                    .stream().map(PitchInfo::getInstrumentType).toArray(String[]::new);
            String[] userNotePrediction = drumPredictList.get(noteInfoIdx);
            Arrays.sort(answerNotePrediction);
            Arrays.sort(userNotePrediction);
            if (Arrays.equals(answerNotePrediction, userNotePrediction)) {
                correctNoteCount++;
            }
        }
        return ((double) correctNoteCount/matchedUserOnsetIndices.length)*100;
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
                                double score = getScore(onsetMatchResult, drumPredictResponse.getPredictions(), measureInfo);
                                FinalMeasureResult finalMeasureResult = FinalMeasureResult.builder()
                                        .measureNumber(audioMessageDto.getMeasureNumber())
                                        .onsetMatchResult(onsetMeasureDataBuilder.build().getOnsetMatchResult())
                                        .userDrumPredictList(drumPredictResponse.getPredictions())
                                        .score(score).build();
                                measureScoreManager.saveMeasureScore(audioMessageDto.getIdentifier(), audioMessageDto.getMeasureNumber(), finalMeasureResult);
                            }).subscribe();
                }).subscribe();
    }
}
