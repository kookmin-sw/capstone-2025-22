package com.capstone.response;

import com.capstone.exception.InternalServerException;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.*;

@Builder
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
public class CustomResponseDto<T> {
    private static ObjectMapper mapper = new ObjectMapper();
    private String message;
    private T body;
    public static <T> T resolveBody(String response, Class<T> clazz) {
        try {
            JsonNode root = mapper.readTree(response);
            JsonNode body = root.get("body");
            return mapper.treeToValue(body, clazz);
        }catch (JsonProcessingException e) {
            throw new InternalServerException("failed to parse json response : " + response);
        }
    }
}
