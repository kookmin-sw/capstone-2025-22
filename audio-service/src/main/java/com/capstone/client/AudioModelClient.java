package com.capstone.client;

import com.capstone.dto.UserResponseDto;
import com.capstone.response.CustomResponseDto;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;

@Component
public class AudioModelClient {
    private final WebClient userClient;
    public AudioModelClient(@Qualifier(WebClientConfig.userClientName) final WebClient userClient) {
        this.userClient = userClient;
    }
    public Mono<UserResponseDto> testUserExists(String email) {
        return userClient.get()
                .uri(builder -> builder.path("/users/email")
                        .queryParam("email", email).build())
                .exchangeToMono(response -> {
                    if(response.statusCode().is4xxClientError()) return Mono.empty();
                    return response.bodyToMono(String.class).map(resBody-> CustomResponseDto.resolveBody(resBody, UserResponseDto.class));
                });
    }
}
