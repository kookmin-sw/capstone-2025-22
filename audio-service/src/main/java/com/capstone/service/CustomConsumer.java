package com.capstone.service;

import com.capstone.client.AudioModelClient;
import com.capstone.dto.AudioMessageDto;
import com.capstone.dto.UserResponseDto;
import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.messaging.handler.annotation.Payload;
import org.springframework.messaging.simp.SimpMessagingTemplate;
import org.springframework.stereotype.Service;
import reactor.core.publisher.Mono;

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
        audioModelClient.testUserExists(audioMessageDto.getMessage())
                .switchIfEmpty(Mono.just(new UserResponseDto()))
                .doOnNext(res -> {
                    if(res.getEmail()==null) {
                        audioMessageDto.setMessage("no such user");
                        messagingTemplate.convertAndSend("/topic/audio", audioMessageDto);
                    }
                    else messagingTemplate.convertAndSend("/topic/audio", audioMessageDto);
                })
                .subscribe();
//        messagingTemplate.convertAndSend("/topic/audio", audioMessageDto.toString());
    }
    @KafkaListener(topics = "sheet", groupId = "${spring.kafka.consumer.group-id}")
    public void sendSheetConversionResult(@Payload final AudioMessageDto audioMessageDto) {
        messagingTemplate.convertAndSend("/topic/audio", audioMessageDto);
    }
}
