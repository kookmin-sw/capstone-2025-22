package com.capstone.service;

import com.capstone.client.AudioModelClient;
import com.capstone.client.MusicClientService;
import com.capstone.config.EmbeddedRedisConfig;
import com.capstone.constants.DrumInstrument;
import com.capstone.dto.AudioMessageDto;
import com.capstone.dto.score.OnsetMatchResult;
import com.capstone.dto.musicXml.MeasureInfo;
import com.capstone.dto.musicXml.NoteInfo;
import com.capstone.dto.musicXml.PitchInfo;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.boot.test.mock.mockito.SpyBean;
import org.springframework.context.annotation.Import;
import org.springframework.kafka.test.context.EmbeddedKafka;
import org.springframework.test.context.ActiveProfiles;
import reactor.core.publisher.Mono;

import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;
import static com.capstone.dto.ModelDto.*;
import static com.capstone.dto.sheet.MusicServiceClientDto.*;

@SpringBootTest
@ActiveProfiles({"test", "webclient", "redis"})
@Import(EmbeddedRedisConfig.class)
@EmbeddedKafka(partitions = 1, controlledShutdown = true)
class AudioMessageConsumerTest {
    @SpyBean
    AudioMessageConsumer audioMessageConsumer;

    @MockBean
    private AudioModelClient audioModelClient;

    @MockBean
    private MusicClientService musicClientService;

    List<OnsetMatchResult> onsetMatchResults;
    List<String> userOnset;
    List<String[]> drumPredictList;
    MeasureInfo measureInfo;

    public MeasureInfo getTestMeasureInfo(List<String[]> instrumentTypesList){
        double currentOnset = 0.0;
        List<NoteInfo> noteInfoList = new ArrayList<>();
        for(String[] instrumentTypes : instrumentTypesList){
            List<PitchInfo> pitchInfoList = new ArrayList<>();
            for(String instrumentType : instrumentTypes){
                pitchInfoList.add(PitchInfo.builder()
                        .instrumentType(instrumentType)
                        .build());
            }
            noteInfoList.add(NoteInfo.builder()
                    .pitchList(pitchInfoList)
                    .startOnset(currentOnset)
                    .endOnset(currentOnset+=1)
                    .build());
        }
        return MeasureInfo.builder()
                .noteList(noteInfoList)
                .build();
    }

    @BeforeEach
    void setUp() {
        OnsetMatchResult onsetMatchResultCorrect5 = OnsetMatchResult.builder()
                .userOnset(List.of(1.0, 2.0, 3.0, 4.0, 5.0))
                .answerOnset(List.of(1.0, 2.0, 3.0, 4.0, 5.0))
                .matchedUserOnsetIndices(new int[]{0, 1, 2, 3, 4})
                .answerOnsetPlayed(new boolean[]{true, true, true, true, true})
                .build();
        OnsetMatchResult onsetMatchResultCorrect4 = OnsetMatchResult.builder()
                .userOnset(List.of(1.0, 2.0, 3.0, 4.0, 5.0))
                .answerOnset(List.of(1.0, 2.0, 3.0, 4.5, 5.0))
                .matchedUserOnsetIndices(new int[]{0, 1, 2, -1, 4})
                .answerOnsetPlayed(new boolean[]{true, true, true, true, true})
                .build();
        onsetMatchResults = List.of(onsetMatchResultCorrect5, onsetMatchResultCorrect4);
        userOnset = List.of("0.0", "1.0", "2.0", "3.0", "4.0");
        drumPredictList = List.of(
                new String[]{DrumInstrument.SNARE, DrumInstrument.TOM},
                new String[]{DrumInstrument.TOM, DrumInstrument.SNARE},
                new String[]{DrumInstrument.TOM},
                new String[]{DrumInstrument.SNARE},
                new String[]{DrumInstrument.KICK}
        );
        measureInfo = getTestMeasureInfo(drumPredictList);
    }

    @Test
    void sendAudioConversionResult_notSaveToDatabase_success(){
        // given
        int userSheetId = 1;
        String audioBase64 = "audioBase64";
        String email = "test@test.com";
        String identifier = UUID.randomUUID().toString();
        String measureNumber = "measureNumber";
        AudioMessageDto messageDto = AudioMessageDto.builder()
                .message(audioBase64)
                .bpm(60)
                .email(email)
                .identifier(identifier)
                .userSheetId(userSheetId)
                .measureNumber(measureNumber)
                .endOfMeasure(false).build();
        // stub
        when(musicClientService.getMeasureInfo(userSheetId, measureNumber)).thenReturn(Mono.just(measureInfo));
        when(audioModelClient.getOnsetFromWav(any(OnsetRequestDto.class))).thenReturn(Mono.just(OnsetResponseDto.builder()
                .onsets(userOnset)
                .count(5).build()));
        when(audioModelClient.getDrumPredictions(any(DrumPredictRequest.class))).thenReturn(Mono.just(DrumPredictResponse.builder()
                .predictions(drumPredictList).build()));
        when(musicClientService.saveMeasureScoreInfo(any(SheetPracticeCreateRequest.class))).thenReturn(Mono.just(true));
        // when && then
        assertDoesNotThrow(() -> {
            audioMessageConsumer.sendAudioConversionResult(messageDto);
            verify(musicClientService, times(1)).getMeasureInfo(userSheetId, measureNumber);
            verify(audioModelClient, times(1)).getOnsetFromWav(any(OnsetRequestDto.class));
            verify(audioModelClient, times(1)).getDrumPredictions(any(DrumPredictRequest.class));
            verify(musicClientService, times(0)).saveMeasureScoreInfo(any(SheetPracticeCreateRequest.class));
        });
    }

    @Test
    void sendAudioConversionResult_saveToDatabase_success(){
        // given
        int userSheetId = 1;
        String audioBase64 = "audioBase64";
        String email = "test@test.com";
        String identifier = UUID.randomUUID().toString();
        String measureNumber = "measureNumber";
        AudioMessageDto messageDto = AudioMessageDto.builder()
                .message(audioBase64)
                .bpm(60)
                .email(email)
                .identifier(identifier)
                .userSheetId(userSheetId)
                .measureNumber(measureNumber)
                .endOfMeasure(true).build();
        // stub
        when(musicClientService.getMeasureInfo(userSheetId, measureNumber)).thenReturn(Mono.just(measureInfo));
        when(audioModelClient.getOnsetFromWav(any(OnsetRequestDto.class))).thenReturn(Mono.just(OnsetResponseDto.builder()
                .onsets(userOnset)
                .count(5).build()));
        when(audioModelClient.getDrumPredictions(any(DrumPredictRequest.class))).thenReturn(Mono.just(DrumPredictResponse.builder()
                .predictions(drumPredictList).build()));
        when(musicClientService.saveMeasureScoreInfo(any(SheetPracticeCreateRequest.class))).thenReturn(Mono.just(true));
        // when && then
        assertDoesNotThrow(() -> {
            audioMessageConsumer.sendAudioConversionResult(messageDto);
            verify(musicClientService, times(1)).getMeasureInfo(userSheetId, measureNumber);
            verify(audioModelClient, times(1)).getOnsetFromWav(any(OnsetRequestDto.class));
            verify(audioModelClient, times(1)).getDrumPredictions(any(DrumPredictRequest.class));
            verify(musicClientService, times(1)).saveMeasureScoreInfo(any(SheetPracticeCreateRequest.class));
        });
    }
}