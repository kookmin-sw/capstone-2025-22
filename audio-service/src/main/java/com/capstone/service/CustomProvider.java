package com.capstone.service;

import com.capstone.dto.AudioMessageDto;
import lombok.extern.slf4j.Slf4j;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.kafka.support.SendResult;
import org.springframework.stereotype.Service;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;

@Slf4j
@Service
public class CustomProvider {
    private final KafkaTemplate<String, AudioMessageDto> kafkaTemplate;
    public CustomProvider(KafkaTemplate<String, AudioMessageDto> kafkaTemplate) {
        this.kafkaTemplate = kafkaTemplate;
    }
    public boolean produceAudioEvent(AudioMessageDto audioMessageDto) {
        try {
            CompletableFuture<SendResult<String, AudioMessageDto>> future = kafkaTemplate.send("audio", audioMessageDto);
            String result = future.handle((res, error) -> {
                if(error != null) {
                    log.error(error.getMessage());
                    return null;
                }
                log.info(audioMessageDto.toString());
                return audioMessageDto.toString();
            }).get();
            return result != null;
        }catch (InterruptedException | ExecutionException e) {
            log.error(e.getMessage());
            return false;
        }
    }
}
