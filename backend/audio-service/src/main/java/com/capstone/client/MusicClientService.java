package com.capstone.client;

import com.capstone.dto.musicXml.MeasureInfo;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;

@Slf4j
@Component
public class MusicClientService {
    private final WebClient musicWebClient;

    public MusicClientService(@Qualifier(WebClientConfig.musicClientName) WebClient musicWebClient) {
        this.musicWebClient = musicWebClient;
    }

    public Mono<MeasureInfo> getMeasureInfo(int userSheetId, String measureNumber){
        return musicWebClient.get()
                .uri(builder -> builder.path("/sheets/{userSheetId}/measures")
                        .queryParam("measureNumber", measureNumber).build(userSheetId))
                .retrieve()
                .bodyToMono(MeasureInfo.class)
                .doOnNext(measureInfo -> log.info("measureInfo: {}", measureInfo.toString()))
                .doOnError(e -> log.error("Error getting measure info: {}", e.getMessage()));
    }
}
