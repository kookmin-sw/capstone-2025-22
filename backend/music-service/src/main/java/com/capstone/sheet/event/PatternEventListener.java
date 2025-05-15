package com.capstone.sheet.event;

import com.capstone.sheet.dto.PatternCreateDto;
import com.capstone.sheet.service.PatternManageService;
import lombok.RequiredArgsConstructor;
import org.springframework.context.event.EventListener;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Component;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.multipart.MultipartFile;

@Component
@RequiredArgsConstructor
public class PatternEventListener {

    private final PatternManageService patternManageService;

    @Async
    @EventListener
    @Transactional
    public void handleCreatePatternEvent(PatternCreateEvent event){
        patternManageService.savePattern(event.patternCreateDto, event.sheetFile, event.patternWav);
    }
}
