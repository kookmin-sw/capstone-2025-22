package com.capstone.service;

import com.capstone.client.AudioModelClient;
import com.capstone.client.MusicClientService;
import com.capstone.config.EmbeddedRedisConfig;
import com.capstone.constants.DrumInstrument;
import com.capstone.dto.OnsetMatchResult;
import com.capstone.dto.musicXml.MeasureInfo;
import com.capstone.dto.musicXml.NoteInfo;
import com.capstone.dto.musicXml.PitchInfo;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.context.annotation.Import;
import org.springframework.messaging.simp.SimpMessagingTemplate;
import org.springframework.test.context.ActiveProfiles;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

@SpringBootTest
@ActiveProfiles("test")
@Import(EmbeddedRedisConfig.class)
class AudioMessageConsumerTest {
    @Autowired
    AudioMessageConsumer audioMessageConsumer;

    @MockBean
    SimpMessagingTemplate messagingTemplate;

    @MockBean
    AudioModelClient audioModelClient;

    @MockBean
    MusicClientService musicClientService;

    @MockBean
    MeasureScoreManager measureScoreManager;

    public MeasureInfo getTestMeasureInfo(List<String> instrumentTypes){
        List<PitchInfo> pitchInfos = instrumentTypes
                .stream().map(instrumentType -> PitchInfo.builder()
                        .instrumentType(instrumentType)
                        .build())
                .toList();
        NoteInfo noteInfo = NoteInfo.builder()
                .pitchList(pitchInfos)
                .build();
        return MeasureInfo.builder()
                .noteList(List.of(noteInfo, noteInfo))
                .build();
    }

    @Test
    void getScore() {
        // given
        String[] answerPredictions = new String[]{DrumInstrument.TOM, DrumInstrument.TOM};
        String[] userPredictions = new String[]{DrumInstrument.TOM, DrumInstrument.SNARE};
        List<String> testInstrumentTypes = List.of(userPredictions);
        MeasureInfo testMeasureInfo = getTestMeasureInfo(testInstrumentTypes);
        OnsetMatchResult onsetMatchResult = OnsetMatchResult.builder()
                .matchedUserOnsetIndices(new int[]{0, 1})
                .build();
        List<String[]> drumPredictionListExpected0 = List.of(answerPredictions, answerPredictions);
        List<String[]> drumPredictionListExpected50 = List.of(answerPredictions, userPredictions);
        List<String[]> drumPredictionListExpected100 = List.of(userPredictions, userPredictions);
        // when
        double scoreMust0 = audioMessageConsumer.getScore(onsetMatchResult, drumPredictionListExpected0, testMeasureInfo);
        double scoreMust50 = audioMessageConsumer.getScore(onsetMatchResult, drumPredictionListExpected50, testMeasureInfo);
        double scoreMust100 = audioMessageConsumer.getScore(onsetMatchResult, drumPredictionListExpected100, testMeasureInfo);
        // then
        assertEquals(100.0, scoreMust100);
        assertEquals(50.0, scoreMust50);
        assertEquals(0.0, scoreMust0);
    }
}