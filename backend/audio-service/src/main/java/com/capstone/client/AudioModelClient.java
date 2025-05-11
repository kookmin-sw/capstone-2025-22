package com.capstone.client;

import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;

import javax.print.attribute.standard.Media;

import static com.capstone.dto.ModelDto.*;

@Slf4j
@Component
public class AudioModelClient {
    private final WebClient audioClient;

    public AudioModelClient(
            @Value("${client.model-service-url}") String modelServiceUrl) {
        this.audioClient = WebClient.builder()
                .defaultHeader(HttpHeaders.CONTENT_TYPE, MediaType.APPLICATION_JSON_VALUE)
                .filter((request, next) -> {
                    log.info("[WebClient 요청] {} {}", request.method(), request.url());
                    request.headers().forEach((name, values) ->
                            values.forEach(value -> log.info("Header: {}={}", name, value)));
                    return next.exchange(request);
                })
                .baseUrl(modelServiceUrl)
                .build();
    }

    public Mono<OnsetResponseDto> getOnsetFromWav(OnsetRequestDto requestDto) {
        return audioClient.post()
                .uri("/onset")
                .accept(MediaType.APPLICATION_JSON)
                .bodyValue(requestDto)
                .exchangeToMono(res -> {
                    if(res.statusCode().isError()){
                        return Mono.error(new RuntimeException("Error getting onset from WAV"));
                    }
                    return res.bodyToMono(OnsetResponseDto.class)
                            .flatMap(onset -> {
                                log.info("onset response {}", onset.toString());
                                return Mono.just(onset);
                            });
                });
    }

    public Mono<DrumPredictResponse> getDrumPredictions(DrumPredictRequest requestDto) {
        return audioClient.post()
                .uri("/onset/predict")
                .accept(MediaType.APPLICATION_JSON)
                .bodyValue(requestDto)
                .exchangeToMono(res -> {
                    if (res.statusCode().isError()) {
                        return Mono.error(new RuntimeException("Error getting drum predictions"));
                    }
                    return res.bodyToMono(DrumPredictResponse.class)
                            .flatMap(drumPredictResponse -> {
                                log.info("drum predict response {}", drumPredictResponse.toString());
                                return Mono.just(drumPredictResponse);
                            });
                });
    }
}
