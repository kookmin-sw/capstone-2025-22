package com.capstone.client;

import com.capstone.exception.CustomException;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;

@Service
public class AuthClientService {
    private final WebClient authClient;
    public AuthClientService(@Qualifier("authWebClient") WebClient authWebClient) {
        this.authClient = authWebClient;
    }
    public String findEmailTokenSync(String email) {
        return authClient.get()
                .uri("/email-token")
                .accept(MediaType.APPLICATION_JSON)
                .exchangeToMono(res -> {
                    if(res.statusCode().is2xxSuccessful()) return res.bodyToMono(String.class);
                    else throw new CustomException(HttpStatus.INTERNAL_SERVER_ERROR, "Failed to find email token");
                })
                .block();
    }
}
