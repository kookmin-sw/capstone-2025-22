package com.capstone.sheet.dto.musicXml;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class PitchInfo {
    String defaultX;
    String noteHead;
    String noteType;
    String pitchType;
    String duration;
    String displayStep;
    String displayOctave;
}
