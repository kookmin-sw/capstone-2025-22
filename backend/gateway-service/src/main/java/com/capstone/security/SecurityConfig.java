package com.capstone.security;

import io.swagger.v3.oas.models.servers.Server;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.HttpMethod;
import org.springframework.security.config.annotation.web.reactive.EnableWebFluxSecurity;
import org.springframework.security.config.web.server.SecurityWebFiltersOrder;
import org.springframework.security.config.web.server.ServerHttpSecurity;
import org.springframework.security.web.server.SecurityWebFilterChain;
import org.springframework.web.cors.CorsConfiguration;
import org.springframework.web.cors.reactive.CorsWebFilter;
import org.springframework.web.cors.reactive.UrlBasedCorsConfigurationSource;
import org.springframework.web.server.ServerWebExchange;

import java.util.List;

@Configuration
@EnableWebFluxSecurity
public class SecurityConfig {
    private final String[] WHITE_LIST = {
            // swagger
            "/h2-console/**",
            "/favicon.ico",
            "/error",
            "/swagger-ui/**",
            "/webjars/swagger-ui/**",
            "/swagger-resources/**",
            "/v3/api-docs/**",
            "/auth/v3/api-docs/**",
            "/verification/v3/api-docs/**",
            "/users/v3/api-docs/**",
            "/music/v3/api-docs/**",
            // auth service
            "/auth/**",
            // verification service
            "/verification/**",
            // socket
            "/ws/**",
    };
    private final String[] WHITE_LIST_GET = {
            // music service
            "/sheets/**",
            // user service
            "/users/**",
    };
    private JwtAuthFilter jwtAuthFilter;
    public SecurityConfig(JwtAuthFilter jwtAuthFilter) {
        this.jwtAuthFilter = jwtAuthFilter;
    }
    @Bean
    public SecurityWebFilterChain securityFilterChain(ServerHttpSecurity http){
        http
                .csrf(ServerHttpSecurity.CsrfSpec::disable)
                .cors(ServerHttpSecurity.CorsSpec::disable)
                .formLogin(ServerHttpSecurity.FormLoginSpec::disable)
                .addFilterBefore(jwtAuthFilter, SecurityWebFiltersOrder.AUTHENTICATION)
                .authorizeExchange(exchange -> {
                    exchange.pathMatchers(WHITE_LIST).permitAll();
                    exchange.pathMatchers(HttpMethod.GET ,WHITE_LIST_GET).permitAll();
                    exchange.anyExchange().authenticated();
                });
        return http.build();
    }
    @Bean
    public CorsWebFilter corsWebFilter(){
        UrlBasedCorsConfigurationSource source = new UrlBasedCorsConfigurationSource(){
            @Override
            public CorsConfiguration getCorsConfiguration(ServerWebExchange exchange) {
                String path = exchange.getRequest().getURI().getPath();
                if(path.startsWith("/ws")){
                    return null;
                }
                CorsConfiguration config = new CorsConfiguration();
                config.setAllowedHeaders(List.of("*"));
                config.setExposedHeaders(List.of("*"));
                config.setAllowedOriginPatterns(List.of("*"));
                config.setAllowedMethods(List.of("GET", "POST", "PUT", "DELETE"));
                return config;
            }
        };
        return new CorsWebFilter(source);
    }
}
