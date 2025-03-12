package com.capstone.controller;

import com.capstone.dto.AudioMessageDto;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.messaging.handler.annotation.MessageMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class AudioController {
    KafkaTemplate<String, AudioMessageDto> kafkaTemplate;
    public AudioController(KafkaTemplate<String, AudioMessageDto> kafkaTemplate) {
        this.kafkaTemplate = kafkaTemplate;
    }
    @MessageMapping("/audio/forwarding")
    public void sendAudio(AudioMessageDto audio){
        
    }
}
