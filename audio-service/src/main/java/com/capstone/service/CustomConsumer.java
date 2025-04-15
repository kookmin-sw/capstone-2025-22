package com.capstone.service;

import com.capstone.client.AudioModelClient;
import com.capstone.dto.AudioMessageDto;
import com.capstone.dto.OnsetDto;
import lombok.extern.slf4j.Slf4j;
import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.messaging.handler.annotation.Payload;
import org.springframework.messaging.simp.SimpMessagingTemplate;
import org.springframework.stereotype.Service;

@Slf4j
@Service
public class CustomConsumer {
    private final SimpMessagingTemplate messagingTemplate;
    private final AudioModelClient audioModelClient;
    public CustomConsumer(SimpMessagingTemplate messagingTemplate, AudioModelClient audioModelClient) {
        this.messagingTemplate = messagingTemplate;
        this.audioModelClient = audioModelClient;
    }
    @KafkaListener(topics = "audio", groupId = "${spring.kafka.consumer.group-id}")
    public void sendAudioConversionResult(@Payload final AudioMessageDto audioMessageDto) {
        OnsetDto.OnsetRequestDto requestDto = OnsetDto.OnsetRequestDto.fromMessageDto(audioMessageDto);
        audioModelClient.getOnsetFromWav(requestDto)
                .doOnNext(res -> messagingTemplate.convertAndSend("/topic/onset/" + audioMessageDto.getEmail(), res))
                .subscribe();
    }
    @KafkaListener(topics = "sheet", groupId = "${spring.kafka.consumer.group-id}")
    public void sendSheetConversionResult(@Payload final AudioMessageDto audioMessageDto) {
        messagingTemplate.convertAndSend("/topic/audio", audioMessageDto);
    }
}
