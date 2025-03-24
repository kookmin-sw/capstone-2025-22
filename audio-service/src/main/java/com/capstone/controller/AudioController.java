package com.capstone.controller;

import com.capstone.dto.AudioMessageDto;
import com.capstone.service.CustomProvider;
import lombok.extern.slf4j.Slf4j;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.messaging.handler.annotation.MessageMapping;
import org.springframework.web.bind.annotation.RestController;

@Slf4j
@RestController
public class AudioController {
    private final CustomProvider provider;
    public AudioController(CustomProvider provider) {
        this.provider = provider;
    }
    @MessageMapping("/audio/forwarding")
    public void sendAudio(AudioMessageDto audio){
        log.info("Sending audio message: {}", audio.toString());
        provider.produceAudioEvent(audio);
    }
}
