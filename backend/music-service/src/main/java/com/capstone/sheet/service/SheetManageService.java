package com.capstone.sheet.service;

import com.capstone.sheet.dto.SheetCreateMeta;
import com.capstone.sheet.dto.SheetResponseDto;
import com.capstone.sheet.entity.Sheet;
import com.capstone.sheet.entity.UserSheet;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.multipart.MultipartFile;

@Service
public class SheetManageService {
    private final SheetCreateService sheetCreateService;
    private final SheetUpdateService sheetUpdateService;

    public SheetManageService(SheetCreateService sheetCreateService, SheetUpdateService sheetUpdateService) {
        this.sheetCreateService = sheetCreateService;
        this.sheetUpdateService = sheetUpdateService;
    }

    @Transactional
    public SheetResponseDto saveSheetAndUserSheet(SheetCreateMeta sheetCreateMeta, MultipartFile sheetFile) {
        Sheet sheet = sheetCreateService.saveSheet();
        UserSheet userSheet = sheetCreateService.saveUserSheet(sheetCreateMeta, sheet);
        sheetUpdateService.updateSheetInfo(sheet, sheetCreateMeta, sheetFile);
        return SheetResponseDto.from(userSheet);
    }
}
