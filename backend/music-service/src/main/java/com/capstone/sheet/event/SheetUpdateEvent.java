package com.capstone.sheet.event;

import com.capstone.sheet.dto.SheetCreateMeta;
import com.capstone.sheet.entity.Sheet;
import lombok.Builder;
import lombok.Data;
import lombok.RequiredArgsConstructor;
import org.springframework.web.multipart.MultipartFile;

@Data
@Builder
@RequiredArgsConstructor
public class SheetUpdateEvent {
    private final Sheet sheet;
    private final SheetCreateMeta sheetCreateMeta;
    private final MultipartFile sheetFile;
}
