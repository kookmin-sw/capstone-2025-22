package com.capstone.config;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.reactive.function.client.WebClient;

@Configuration
public class WebClientConfig {
    @Bean
    public WebClient authWebClient(){
        return WebClient.builder()
                .baseUrl("http://AUTH-SERVICE")
                .build();
    }
}
