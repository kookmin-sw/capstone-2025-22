package com.capstone.client;

import com.capstone.dto.UserResponseDto;
import com.capstone.dto.musicXml.MeasureInfo;
import com.capstone.response.CustomResponseDto;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;

import static com.capstone.dto.sheet.MusicServiceClientDto.*;

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
                .exchangeToMono(res -> {
                    if(res.statusCode().is2xxSuccessful()){
                        return res.bodyToMono(String.class).map(resBody-> CustomResponseDto.resolveBody(resBody, MeasureInfo.class));
                    }else{
                        String message = res.bodyToMono(String.class).block();
                        log.error("Error getting measure info: {}", message);
                        return Mono.empty();
    public Mono<MeasureInfo> getPatternMeasureInfo(Long patternId, String measureNumber){
        return musicWebClient.get()
                .uri(builder -> builder.path("/patterns/{patternId}/measures")
                        .queryParam("measureNumber", measureNumber).build(patternId))
                .exchangeToMono(res -> {
                    if(res.statusCode().is2xxSuccessful()){
                        return res.bodyToMono(String.class).map(resBody-> CustomResponseDto.resolveBody(resBody, MeasureInfo.class));
                    }else{
                        return res.bodyToMono(String.class)
                                .doOnNext(message -> log.error("[pattern practice] Error getting measure info: {}", message))
                                .then(Mono.empty());
                    }
                });
    }

    public Mono<Boolean> saveMeasureScoreInfo(SheetPracticeCreateRequest requestDto) {
        return musicWebClient.post()
                .uri(builder -> builder.path("/sheets/{userSheetId}/practices")
                        .build(requestDto.getUserSheetId()))
                .exchangeToMono(res -> {
                    if (res.statusCode().is2xxSuccessful()) {
                        return Mono.just(true);
                    }else{
                        String message = res.bodyToMono(String.class).block();
                        log.error("Error saving practice info: {}", message);
                        return Mono.just(false);
    public Mono<Boolean> savePatternScoreInfo(PatternPracticeCreateRequest createDto){
        return musicWebClient.post()
                .uri(builder -> builder.path("patterns/practices").build())
                .bodyValue(createDto)
                .exchangeToMono(res -> {
                    if(res.statusCode().is2xxSuccessful()){
                        return Mono.just(true);
                    }else{
                        return res.bodyToMono(String.class)
                                .doOnNext(message -> log.error("[pattern practice] Error saving practice info: {}", message))
                                .then(Mono.just(false));
                    }
                });
    }
}
