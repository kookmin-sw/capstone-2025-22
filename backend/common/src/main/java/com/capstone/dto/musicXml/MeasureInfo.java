package com.capstone.dto.musicXml;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class MeasureInfo {
    String measureNumber;
    List<NoteInfo> noteList;
}
