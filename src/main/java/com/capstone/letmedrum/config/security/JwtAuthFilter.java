package com.capstone.letmedrum.config.security;

import com.capstone.letmedrum.user.service.CustomUserDetailService;
import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;

import java.io.IOException;

@Slf4j
@Component
public class JwtAuthFilter extends OncePerRequestFilter {
    private final JwtUtils jwtUtils;
    private final CustomUserDetailService customUserDetailService;
    public JwtAuthFilter(JwtUtils jwtUtils, CustomUserDetailService customUserDetailService){
        this.jwtUtils = jwtUtils;
        this.customUserDetailService = customUserDetailService;
    }
    /*
    * 여기서 예외를 던져버리면 permitAll()로 허용해준 경로에서도 예외가 발생할 수 있음
    * => 따라서, Authentication 을 설정하지 않고 넘어가고(미인증 상태로 pass),
    * 인증여부에 따라서 AuthenticationFilter 에서 추후 인증 실패 예외를 처리해줌
    * */
    @Override
    protected void doFilterInternal(
            @NonNull  HttpServletRequest request,
            @NonNull HttpServletResponse response,
            @NonNull FilterChain filterChain) throws ServletException, IOException {
        String authentication = jwtUtils.getTokenFromRequest(request);
        if(authentication!=null && jwtUtils.validateToken(authentication)){
            // get email and principal( = UserDetails )
            String email = jwtUtils.getUserEmail(authentication);
            UserDetails userDetails = email!=null ?
                    customUserDetailService.loadUserByUsername(email) : null;
            if(userDetails==null){
                log.info(String.format("JwtAuthFilter : UserDetails is null (email:%s)", email));
                filterChain.doFilter(request, response);
                return;
            }
            // generate Authentication (UsernamePasswordAuthenticationToken)
            UsernamePasswordAuthenticationToken usernamePasswordAuthenticationToken
                    = new UsernamePasswordAuthenticationToken(
                    userDetails, null, userDetails.getAuthorities());
            // set Authentication on SecurityContextHolder
            SecurityContextHolder
                    .getContext()
                    .setAuthentication(usernamePasswordAuthenticationToken);
        }
        filterChain.doFilter(request, response);
    }
}
