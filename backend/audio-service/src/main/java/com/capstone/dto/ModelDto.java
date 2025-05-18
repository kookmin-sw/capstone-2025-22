package com.capstone.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

@Data
public class ModelDto {
    @Data
    @AllArgsConstructor
    public static class OnsetRequestDto{
        String audio_base64;

        public static OnsetRequestDto fromMessageDto(AudioMessageDto dto){
            return new OnsetRequestDto(dto.getMessage());
        }

        public static OnsetRequestDto fromPatternMessage(PatternMessageDto dto){
            return new OnsetRequestDto(dto.getAudioBase64());
        }
    }

    @Data
    @Builder
    @AllArgsConstructor
    @NoArgsConstructor
    public static class OnsetResponseDto{
        List<String> onsets;
        int count;
    }

    @Data
    @Builder
    @AllArgsConstructor
    @NoArgsConstructor
    public static class DrumPredictRequest{
        String audio_base64;
        List<String> onsets;
    }

    @Data
    @Builder
    @AllArgsConstructor
    @NoArgsConstructor
    public static class DrumPredictResponse{
        List<String[]> predictions;
    }
}
