package com.capstone.dto.musicXml;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class MusicXmlMetaData {
    int bpm;
    int division;
    int beat;
    int beatType;
}
