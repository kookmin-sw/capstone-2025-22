package com.capstone.config;

import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.cors.CorsConfiguration;
import org.springframework.web.cors.reactive.CorsWebFilter;
import org.springframework.web.cors.reactive.UrlBasedCorsConfigurationSource;

import java.util.List;

@Slf4j
@Configuration
public class SecurityConfig {
    @Value("${client.gateway-service-url}")
    private String gatewayUrl;
    @Value("${gateway.service-url}")
    private String externalGatewayUrl;
    @Bean
    public CorsWebFilter corsWebFilter() {
        log.info("allowed urls: {}, {}", gatewayUrl, externalGatewayUrl);
        CorsConfiguration config = new CorsConfiguration();
        config.setAllowedHeaders(List.of("*"));
        config.setExposedHeaders(List.of("*"));
//        config.setAllowedOrigins(List.of(gatewayUrl, externalGatewayUrl));
        config.setAllowedOrigins(List.of("*"));
        config.setAllowedMethods(List.of("GET", "POST", "PUT", "DELETE"));
        config.setAllowCredentials(true);
        UrlBasedCorsConfigurationSource source = new UrlBasedCorsConfigurationSource();
        source.registerCorsConfiguration("/**", config);
        return new CorsWebFilter(source);
    }
}

