package com.capstone.jwt;

import com.capstone.auth.UserAuthInfoDto;
import com.capstone.constants.AuthConstants;
import com.capstone.enums.UserRole;
import io.jsonwebtoken.Claims;
import io.jsonwebtoken.Jwts;
import io.jsonwebtoken.io.Decoders;
import io.jsonwebtoken.security.Keys;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpRequest;
import org.springframework.stereotype.Component;

import java.security.Key;
import java.time.ZoneId;
import java.time.ZonedDateTime;
import java.util.Date;

@Component
@Slf4j
public class JwtUtils {
    private final Key SECRET_KEY;
    public JwtUtils(@Value("${jwt.secret-key}") String secret_key){
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
        return this.generateJwtToken(dto, zoneId, AuthConstants.ACCESS_TOKEN_EXP_TIME);
    }
    /**
     * generate refresh token
     * @param dto UserAuthInfoDto(email, role)
     * @return refreshToken - String, not-null
     * */
    public String generateRefreshToken(UserAuthInfoDto dto){
        ZoneId zoneId = ZoneId.systemDefault();
        return this.generateJwtToken(dto, zoneId, AuthConstants.REFRESH_TOKEN_EXP_TIME);
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
        if(token.startsWith(AuthConstants.ACCESS_TOKEN_PREFIX)){
            return token.substring(AuthConstants.ACCESS_TOKEN_PREFIX.length());
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
            log.info("claim is null (jwtUtils.getUserEmail)");
            return null;
        }
        return claims.get("email", String.class);
    }
    /**
     * @param token jwtToken String
     * @return role - UserRole, nullable
     * */
    public UserRole getUserRole(String token){
        Claims claims = getClaims(token);
        if(claims == null){
            log.info("claim is null (jwtUtils.getUserRole)");
            return null;
        }
        return claims.get("role", UserRole.class);
    }
    /**
     * get access token from HttpServletRequest object
     * @param request HttpServletRequest
     * @return accessToken - String, nullable */
    public String getTokenFromRequest(HttpRequest request){
        try {
            String authorization = request.getHeaders().get(AuthConstants.ACCESS_TOKEN_HEADER_KEY).get(0);
            if(authorization==null || !authorization.startsWith(AuthConstants.ACCESS_TOKEN_PREFIX)){
                log.info("JwtUtils : authentication is null or invalid prefix : {}", authorization);
                // logRequestHeader(request);
                return null;
            }
            return authorization.substring(AuthConstants.ACCESS_TOKEN_PREFIX.length());
        }catch (NullPointerException e){
            return null;
        }
    }
    /**
     * logging request header
     * @param request HttpServletRequest */
    public void logRequestHeader(HttpRequest request){
        HttpHeaders headers = request.getHeaders();  // HttpHeaders 객체를 가져옴
        headers.forEach((headerName, headerValues) -> {
            headerValues.forEach(headerValue -> {
                log.info("header => {}: {}", headerName, headerValue);
            });
        });
    }
}
