package com.capstone.sheet.event;

import com.capstone.sheet.dto.PatternCreateDto;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.springframework.web.multipart.MultipartFile;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class PatternCreateEvent {
    PatternCreateDto patternCreateDto;
    byte[] sheetFile;
    byte[] patternWav;
}
