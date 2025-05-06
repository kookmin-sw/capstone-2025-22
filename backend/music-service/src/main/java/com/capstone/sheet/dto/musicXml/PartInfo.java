package com.capstone.sheet.dto.musicXml;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class PartInfo {
    int beats;
    int beatType;
    int divisions;
    List<MeasureInfo> measureList;
}
