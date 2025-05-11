package com.capstone.client;

import org.apache.http.HttpHeaders;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Profile;
import org.springframework.http.MediaType;
import org.springframework.web.reactive.function.client.WebClient;

@Configuration
@Profile("webclient")
public class WebClientConfig {
    public static final String userClientName = "userClient";
    public static final String authClientName = "authClient";
    public static final String verificationClientName = "verificationClient";
    public static final String googleClientName = "googleClient";
    public static final String musicClientName = "musicClient";
    private final String userBaseUrl;
    private final String authBaseUrl;
    private final String verificationBaseUrl;
    private final String musicBaseUrl;
    public WebClientConfig(
            @Value("${client.user-service-url}") String userBaseUrl,
            @Value("${client.auth-service-url}") String authBaseUrl,
            @Value("${client.verification-service-url}") String verificationBaseUrl,
            @Value("${client.music-service-url}") String musicBaseUrl){
        this.userBaseUrl = userBaseUrl;
        this.authBaseUrl = authBaseUrl;
        this.verificationBaseUrl = verificationBaseUrl;
        this.musicBaseUrl = musicBaseUrl;
    }
    @Bean
    public WebClient userClient(WebClient.Builder webClientBuilder) {
        return webClientBuilder
                .baseUrl(userBaseUrl)
                .build();
    }
    @Bean
    public WebClient authClient(WebClient.Builder webClientBuilder) {
        return webClientBuilder
                .baseUrl(authBaseUrl)
                .build();
    }
    @Bean
    public WebClient verificationClient(WebClient.Builder webClientBuilder) {
        return webClientBuilder
                .baseUrl(verificationBaseUrl)
                .build();
    }
    @Bean
    public WebClient musicClient(WebClient.Builder webClientBuilder) {
        return webClientBuilder
                .baseUrl(musicBaseUrl)
                .build();
    }
    @Bean
    public WebClient googleClient(){
        return WebClient.builder()
                .defaultHeader(HttpHeaders.CONTENT_TYPE, MediaType.APPLICATION_JSON_VALUE)
                .build();
    }
}
