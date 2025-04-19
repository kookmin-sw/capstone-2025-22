package com.capstone.dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

@Data
public class OnsetDto {
    @Data
    @AllArgsConstructor
    public static class OnsetRequestDto{
        String audio_base64;

        public static OnsetRequestDto fromMessageDto(AudioMessageDto dto){
            return new OnsetRequestDto(dto.getMessage());
        }
    }

    @Data
    @AllArgsConstructor
    @NoArgsConstructor
    public static class OnsetResponseDto{
        List<String> onsets;
        int count;
    }
}
