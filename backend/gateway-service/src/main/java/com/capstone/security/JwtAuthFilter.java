package com.capstone.security;

import com.capstone.jwt.JwtUtils;
import lombok.extern.slf4j.Slf4j;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.context.ReactiveSecurityContextHolder;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.stereotype.Component;
import org.springframework.web.server.ServerWebExchange;
import org.springframework.web.server.WebFilter;
import org.springframework.web.server.WebFilterChain;
import reactor.core.publisher.Mono;

@Slf4j
@Component
public class JwtAuthFilter implements WebFilter {
    private final JwtUtils jwtUtils;
    private final CustomUserDetailService customUserDetailService;
    public JwtAuthFilter(JwtUtils jwtUtils, CustomUserDetailService customUserDetailService){
        this.jwtUtils = jwtUtils;
        this.customUserDetailService = customUserDetailService;
    }
    /**
     * jwt 인증 필터
     * @param exchange 사용자 요청 추출에 사용할 객체
     * @param chain webFilterChain
    * */
    @Override
    public Mono<Void> filter(ServerWebExchange exchange, WebFilterChain chain){
        return Mono.justOrEmpty(exchange.getRequest())
                .flatMap(req -> {
                    String token = jwtUtils.getTokenFromRequest(req);
                    if(token == null || !jwtUtils.validateToken(token)){
                        log.info("Invalid token : {}", token);
                        return chain.filter(exchange);
                    }
                    UserDetails userDetails = customUserDetailService.loadUserByUsername(jwtUtils.getUserEmail(token));
                    UsernamePasswordAuthenticationToken authentication
                            = new UsernamePasswordAuthenticationToken(userDetails, null, userDetails.getAuthorities());
                    return chain.filter(exchange)
                            .contextWrite(ReactiveSecurityContextHolder.withAuthentication(authentication));
                });
    }
}
