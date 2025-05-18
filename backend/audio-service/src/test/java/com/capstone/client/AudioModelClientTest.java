package com.capstone.client;

import com.capstone.config.EmbeddedKafkaConfig;
import com.capstone.config.EmbeddedRedisConfig;
import com.capstone.constants.DrumInstrument;
import com.capstone.dto.ModelDto;
import com.capstone.service.MeasureScoreManager;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.context.annotation.Import;
import org.springframework.core.io.ClassPathResource;
import org.springframework.kafka.test.context.EmbeddedKafka;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.test.web.servlet.MockMvc;
import reactor.core.publisher.Mono;

import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

@SpringBootTest
@ActiveProfiles({"test", "webclient", "redis"})
@AutoConfigureMockMvc
@Import({EmbeddedKafkaConfig.class, EmbeddedRedisConfig.class})
class AudioModelClientTest {
    @MockBean
    AudioModelClient audioModelClient;

    @Autowired
    MockMvc mockMvc;

    @MockBean
    MeasureScoreManager measureScoreManager;

    @Test
    void getDrumPredictions_test() throws Exception{
        // given
        String audioBase64 = Files.readString(Paths.get(new ClassPathResource("audio_base64.txt").getURI()));
        ModelDto.DrumPredictRequest requestDto = ModelDto.DrumPredictRequest.builder()
                .audio_base64(audioBase64)
                .onsets(List.of("0.17414965986394557",
                        "1.555736961451247",
                        "2.809614512471655",
                        "3.93578231292517",
                        "5.3986394557823125"))
                .build();
        // stub
        when(audioModelClient.getDrumPredictions(any(ModelDto.DrumPredictRequest.class)))
                .thenReturn(Mono.just(ModelDto.DrumPredictResponse.builder()
                        .predictions(List.of(
                                new String[]{DrumInstrument.TOM},
                                new String[]{DrumInstrument.TOM},
                                new String[]{DrumInstrument.TOM},
                                new String[]{DrumInstrument.TOM},
                                new String[]{DrumInstrument.TOM}
                        )).build()));
        // when
        ModelDto.DrumPredictResponse res = audioModelClient.getDrumPredictions(requestDto).block();
        // then
        assert res!=null && res.getPredictions().size() == 5;
    }
}