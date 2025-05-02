package com.capstone.sheet.service;

import com.capstone.sheet.dto.SheetCreateMeta;
import com.capstone.sheet.entity.Sheet;
import com.capstone.sheet.entity.UserSheet;
import com.capstone.sheet.repository.SheetRepository;
import com.capstone.sheet.repository.UserSheetRepository;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
public class SheetCreateService {
    private final SheetRepository sheetRepository;
    private final UserSheetRepository userSheetRepository;

    public SheetCreateService(
            SheetRepository sheetRepository,
            UserSheetRepository userSheetRepository) {
        this.sheetRepository = sheetRepository;
        this.userSheetRepository = userSheetRepository;
    }

    /**
     * 악보정보 저장
     * @return Sheet
     * */
    @Transactional
    public Sheet saveSheet() {
        return sheetRepository.saveAndFlush(Sheet.create());
    }

    @Transactional
    public UserSheet saveUserSheet(SheetCreateMeta sheetCreateMeta, Sheet sheet){
        return userSheetRepository.saveAndFlush(sheetCreateMeta.toUserSheetEntity(sheet));
    }
}
