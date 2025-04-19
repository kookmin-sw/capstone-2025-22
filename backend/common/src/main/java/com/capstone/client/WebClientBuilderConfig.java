package com.capstone.client;

import lombok.extern.slf4j.Slf4j;
import org.springframework.cloud.client.loadbalancer.LoadBalanced;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Profile;
import org.springframework.web.reactive.function.client.WebClient;

@Slf4j
@Configuration
@Profile("webclient")
public class WebClientBuilderConfig {
    @Bean
    @LoadBalanced
    public WebClient.Builder webClientBuilder() {
        return WebClient.builder()
                .filter((req, next) -> {
                    log.info("Request: [{}] {}", req.method(), req.url());
                    return next.exchange(req).doOnNext(res -> log.info("Received Response : {}", res));
                });
    }
}
