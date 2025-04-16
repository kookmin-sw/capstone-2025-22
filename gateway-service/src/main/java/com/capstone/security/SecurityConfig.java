package com.capstone.security;

import io.netty.handler.codec.http.cors.CorsConfig;
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
            "/h2-console/**",
            "/favicon.ico",
            "/error",
            "/swagger-ui/**",
            "/webjars/swagger-ui/**",
            "/swagger-resources/**",
            "/v3/api-docs/**",
            "/users/v3/api-docs",
            "/music/v3/api-docs",
            "/ws/**",
    };
    private final String[] USER_WHITE_LIST = {
            "/users/nickname",
            "/users/email",
            "/users/password",
    };
    private final String[] VERIFICATION_WHITE_LIST = {
            "/verification/**"
    };
    private final String[] AUTH_WHITE_LIST = {
            "/auth/**"
    };
    private final String[] MUSIC_WHITE_LIST = {
            "/sheets/**",
    };
    private final JwtAuthFilter jwtAuthFilter;
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
                    exchange.pathMatchers(USER_WHITE_LIST).permitAll();
                    exchange.pathMatchers(VERIFICATION_WHITE_LIST).permitAll();
                    exchange.pathMatchers(AUTH_WHITE_LIST).permitAll();
                    exchange.pathMatchers(HttpMethod.GET, MUSIC_WHITE_LIST).permitAll();
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
