package com.capstone.letmedrum.config.security;

import com.capstone.letmedrum.auth.dto.UserAuthInfoDto;
import io.jsonwebtoken.Claims;
import io.jsonwebtoken.Jwts;
import io.jsonwebtoken.io.Decoders;
import io.jsonwebtoken.security.Keys;
import jakarta.servlet.http.HttpServletRequest;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import java.security.Key;
import java.time.ZoneId;
import java.time.ZonedDateTime;
import java.util.Date;
import java.util.Enumeration;

@Component
@Slf4j
public class JwtUtils {
    private final Key SECRET_KEY;
    public static final Long ACCESS_TOKEN_EXP_TIME = 1000000L;
    public static final Long REFRESH_TOKEN_EXP_TIME = 1000000000L;
    public static final String ACCESS_TOKEN_HEADER_KEY = "authorization";
    public static final String ACCESS_TOKEN_PREFIX = "Bearer ";
    public JwtUtils(@Value("${env.jwt.secret-key}") String secret_key){
        byte[] keyBytes = Decoders.BASE64.decode(secret_key);
        this.SECRET_KEY = Keys.hmacShaKeyFor(keyBytes);
    }
    /**
     * generate jwt token
     * @param dto UserAuthInfoDto(email, role)
     * @param zoneId timeZone ZoneId
     * @return JwtToken - String, not-null
     */
    public String generateJwtToken(UserAuthInfoDto dto, ZoneId zoneId, Long exp_time){
        Claims claims = Jwts.claims();
        claims.put("email", dto.getEmail());
        claims.put("role", dto.getRole().name());
        ZonedDateTime issuedDate = ZonedDateTime.now(zoneId);
        ZonedDateTime expireDate = issuedDate.plusSeconds(exp_time);
        return Jwts.builder()
                .setClaims(claims)
                .setIssuedAt(Date.from(issuedDate.toInstant()))
                .setExpiration(Date.from(expireDate.toInstant()))
                .signWith(this.SECRET_KEY)
                .compact();
    }
    /**
     * generate access token
     * @param dto UserAuthInfoDto(email, role)
     * @return accessToken - String, not-null
    */
    public String generateAccessToken(UserAuthInfoDto dto){
        ZoneId zoneId = ZoneId.systemDefault();
        return this.generateJwtToken(dto, zoneId, ACCESS_TOKEN_EXP_TIME);
    }
    /**
     * generate refresh token
     * @param dto UserAuthInfoDto(email, role)
     * @return refreshToken - String, not-null
     * */
    public String generateRefreshToken(UserAuthInfoDto dto){
        ZoneId zoneId = ZoneId.systemDefault();
        return this.generateJwtToken(dto, zoneId, REFRESH_TOKEN_EXP_TIME);
    }
    /**
     * validate token
     * @param token jwtToken String
     * @return isValidate - boolean
     * */
    public boolean validateToken(String token){
        try {
            Jwts
                    .parserBuilder()
                    .setSigningKey(this.SECRET_KEY)
                    .build()
                    .parseClaimsJws(token);
            return true;
        }catch (Exception e){
            System.out.println(e.getMessage());
        }
        return false;
    }
    /**
     * just process token (this function doesn't validate token)
     * @param token access token or refresh token
     * @return preprocessed token
    * */
    public String processToken(String token){
        if(token.startsWith(ACCESS_TOKEN_PREFIX)){
            return token.substring(ACCESS_TOKEN_PREFIX.length());
        }
        return token;
    }
    /**
     * @param token jwtToken String
     * @return JwtToken's Claims - Claims, nullable
     * */
    private Claims getClaims(String token){
        try {
            return Jwts
                    .parserBuilder()
                    .setSigningKey(this.SECRET_KEY)
                    .build()
                    .parseClaimsJws(token)
                    .getBody();
        }catch (Exception e){
            return null;
        }
    }
    /**
     * @param token jwtToken String
     * @return userEmail - String, nullable
     * */
    public String getUserEmail(String token){
        Claims claims = getClaims(token);
        if(claims == null){
            log.info("JwtUtils : Claims object is null on token");
            return null;
        }
        return claims.get("email", String.class);
    }
    /**
     * get access token from HttpServletRequest object
     * @param request HttpServletRequest
     * @return accessToken - String, nullable */
    public String getTokenFromRequest(HttpServletRequest request){
        String authorization = request.getHeader(ACCESS_TOKEN_HEADER_KEY);
        if(authorization==null || !authorization.startsWith(ACCESS_TOKEN_PREFIX)){
            log.info(String.format("JwtUtils : authentication is null or invalid prefix : %s", authorization));
            // logRequestHeader(request);
            return null;
        }
        return authorization.substring(ACCESS_TOKEN_PREFIX.length());
    }
    /**
     * logging request header
     * @param request HttpServletRequest */
    public void logRequestHeader(HttpServletRequest request){
        Enumeration<String> headerNames = request.getHeaderNames();
        if (headerNames != null) {
            while (headerNames.hasMoreElements()) {
                String headerName = headerNames.nextElement();
                String headerValue = request.getHeader(headerName);
                log.info("header => " + headerName + ": " + headerValue);
            }
        }
    }
}
