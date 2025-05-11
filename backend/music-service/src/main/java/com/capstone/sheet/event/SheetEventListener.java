package com.capstone.sheet.event;

import com.capstone.sheet.service.SheetUpdateService;
import lombok.RequiredArgsConstructor;
import org.springframework.context.event.EventListener;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Component;
import org.springframework.transaction.annotation.Transactional;

@Component
@RequiredArgsConstructor
public class SheetEventListener {
    private final SheetUpdateService sheetUpdateService;

    @Async
    @EventListener
    @Transactional
    public void handleSheetInfoUpdateEvent(SheetUpdateEvent event) {
        sheetUpdateService.updateSheetInfo(event.getSheet(), event.getSheetCreateMeta(), event.getSheetFileBytes());
    }
}
