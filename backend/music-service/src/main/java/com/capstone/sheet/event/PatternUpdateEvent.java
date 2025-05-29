package com.capstone.sheet.event;

import com.capstone.sheet.dto.PatternUpdateDto;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class PatternUpdateEvent {
    Long patternId;
    PatternUpdateDto patternUpdateDto;
    byte[] sheetFile;
    byte[] patternWav;
}
