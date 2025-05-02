package com.capstone.sheet.service;

import com.capstone.sheet.dto.SheetCreateMeta;
import com.capstone.sheet.dto.SheetResponseDto;
import com.capstone.sheet.entity.Sheet;
import com.capstone.sheet.entity.UserSheet;
import com.capstone.sheet.event.SheetUpdateEvent;
import lombok.RequiredArgsConstructor;
import org.springframework.context.ApplicationEventPublisher;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.multipart.MultipartFile;

@Service
@RequiredArgsConstructor
public class SheetManageService {
    private final SheetCreateService sheetCreateService;
    private final SheetUpdateService sheetUpdateService;
    private final ApplicationEventPublisher eventPublisher;

    @Transactional
    public SheetResponseDto saveSheetAndUserSheet(SheetCreateMeta sheetCreateMeta, MultipartFile sheetFile) {
        Sheet sheet = sheetCreateService.saveSheet();
        UserSheet userSheet = sheetCreateService.saveUserSheet(sheetCreateMeta, sheet);
        eventPublisher.publishEvent(SheetUpdateEvent.builder()
                        .sheet(sheet)
                        .sheetCreateMeta(sheetCreateMeta)
                        .sheetFile(sheetFile).build());
        return SheetResponseDto.from(userSheet);
    }
}
