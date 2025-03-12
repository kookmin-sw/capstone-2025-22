package com.capstone.controller;

import com.capstone.dto.AudioMessageDto;
import org.springframework.messaging.handler.annotation.MessageMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class AudioController {
    @MessageMapping("/audio/forwarding")
    public void sendAudio(AudioMessageDto audio){
        
    }
}
