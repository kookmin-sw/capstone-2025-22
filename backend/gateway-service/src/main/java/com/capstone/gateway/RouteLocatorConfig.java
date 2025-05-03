package com.capstone.gateway;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.cloud.context.config.annotation.RefreshScope;
import org.springframework.cloud.gateway.route.RouteLocator;
import org.springframework.cloud.gateway.route.builder.RouteLocatorBuilder;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
@RefreshScope
public class RouteLocatorConfig {
    @Value("${client.user-service-url}")
    private String userServiceUrl;
    @Value("${client.auth-service-url}")
    private String authServiceUrl;
    @Value("${client.verification-service-url}")
    private String verificationServiceUrl;
    @Value("${client.music-service-url}")
    private String musicServiceUrl;
    @Value("${client.audio-service-url}")
    private String audioServiceUrl;
    @Bean
    public RouteLocator routeLocator(RouteLocatorBuilder builder) {
        return builder.routes()
                .route("user-service", r -> r.path("/users/**").uri(userServiceUrl))
                .route("auth-service", r -> r.path("/auth/**").uri(authServiceUrl))
                .route("verification-service", r -> r.path("/verification/**").uri(verificationServiceUrl))
                .route("music-service", r-> r.path("/sheets/**", "/music/**").uri(musicServiceUrl))
                .route("audio-service", r-> r.path("/audio/**", "/ws/**").uri(audioServiceUrl))
                .build();
    }
}
