package com.capstone.service;

import com.capstone.client.AudioModelClient;
import com.capstone.client.MusicClientService;
import com.capstone.config.EmbeddedRedisConfig;
import com.capstone.constants.DrumInstrument;
import com.capstone.dto.score.OnsetMatchResult;
import com.capstone.dto.musicXml.MeasureInfo;
import com.capstone.dto.musicXml.NoteInfo;
import com.capstone.dto.musicXml.PitchInfo;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
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
    void calculateScore() {
        // given
        List<Boolean> mustBe100 = List.of(true, true, true, true, true, true);
        List<Boolean> mustBe50 = List.of(true, true, true, false, false, false);
        List<Boolean> mustBe0 = List.of(false, false, false, false, false, false);
        // when
        double scoreMust0 = audioMessageConsumer.calculateScore(mustBe0);
        double scoreMust50 = audioMessageConsumer.calculateScore(mustBe50);
        double scoreMust100 = audioMessageConsumer.calculateScore(mustBe100);
        // then
        assertEquals(100.0, scoreMust100);
        assertEquals(50.0, scoreMust50);
        assertEquals(0.0, scoreMust0);
    }
}