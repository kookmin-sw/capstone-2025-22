package com.capstone.service;

import com.capstone.client.AudioModelClient;
import com.capstone.client.MusicClientService;
import com.capstone.dto.score.FinalMeasureResult;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.test.context.ActiveProfiles;

import java.util.List;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.*;

@SpringBootTest
@ActiveProfiles({"test", "redis"})
class MeasureScoreManagerTest {
    @Autowired
    private MeasureScoreManager measureScoreManager;

    @Autowired
    private ObjectMapper objectMapper;

    @MockBean
    private MusicClientService musicClientService;

    @MockBean
    private AudioModelClient audioModelClient;

    @Test
    @DisplayName("연습 정보 저장 성공 테스트")
    void saveMeasureScore_success() throws Exception {
        // given
        String identifier = UUID.randomUUID().toString();
        String measureNumber = UUID.randomUUID().toString();
        FinalMeasureResult finalMeasureResult = FinalMeasureResult.builder()
                .score(100.0)
                .build();
        // when
        boolean res = measureScoreManager.saveMeasureScore(identifier, measureNumber, finalMeasureResult);
        String savedScore = measureScoreManager.getMeasureScore(identifier, measureNumber);
        // then
        assert res;
        assertEquals(savedScore, finalMeasureResult.toString());
        assert FinalMeasureResult.fromString(savedScore).getScore() == 100.0;
    }

    @Test
    @DisplayName("연습 정보 일괄 조회 성공 테스트")
    void getAllMeasureScores_success() {
        // given
        String identifier = UUID.randomUUID().toString();
        for(int i = 0; i < 10; i++){
            String measureNumber = UUID.randomUUID().toString();
            FinalMeasureResult finalMeasureResult = FinalMeasureResult.builder()
                    .finalScoringResults(List.of(true, true, true))
                    .beatScoringResults(List.of(true, true, true))
                    .measureNumber(measureNumber)
                    .score(((double) i + 1 ) * 10)
                    .build();
            measureScoreManager.saveMeasureScore(identifier, measureNumber, finalMeasureResult);
        }
        // when
        List<String> res = measureScoreManager.getAllMeasureScores(identifier);
        assert res.size() == 10;
    }
}