package com.capstone.service;

import com.capstone.dto.AudioMessageDto;
import com.capstone.dto.PatternMessageDto;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.kafka.support.SendResult;
import org.springframework.stereotype.Service;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;

@Slf4j
@Service
@RequiredArgsConstructor
public class CustomProvider {
    private final KafkaTemplate<String, AudioMessageDto> kafkaTemplate;
    private final KafkaTemplate<String, PatternMessageDto> patternKafkaTemplate;

    public boolean produceAudioEvent(AudioMessageDto audioMessageDto) {
        try {
            CompletableFuture<SendResult<String, AudioMessageDto>> future = kafkaTemplate.send("audio", audioMessageDto);
            String result = future.handle((res, error) -> {
                if(error != null) {
                    log.error(error.getMessage());
                    return null;
                }
                return audioMessageDto.toString();
            }).get();
            return result != null;
        }catch (InterruptedException | ExecutionException e) {
            log.error(e.getMessage());
            return false;
        }
    }

    public void producePatternEvent(PatternMessageDto patternMessageDto) {
        patternKafkaTemplate.send("pattern", patternMessageDto);
    }
}
