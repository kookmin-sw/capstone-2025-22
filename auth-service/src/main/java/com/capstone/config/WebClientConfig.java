package com.capstone.config;

import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.cloud.client.loadbalancer.LoadBalanced;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.reactive.function.client.WebClient;

@Slf4j
@Configuration
public class WebClientConfig{
    @Bean
    @LoadBalanced
    public WebClient.Builder webClientBuilder() {
        return WebClient.builder()
                .filter((req, next) -> {
                    log.info("Request: [{}] {}", req.method(), req.url());
                    return next.exchange(req).doOnNext(res -> {
                        log.info("Received Response : {}", res.toString());
                    });
                });
    }
    @Bean
    public WebClient userWebClient(
            WebClient.Builder webClientBuilder,
            @Value("${client.user-service-url}") String userServiceUrl) {
        return webClientBuilder
                .baseUrl(userServiceUrl)
                .build();
    }
}
