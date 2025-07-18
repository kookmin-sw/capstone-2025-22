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
public class NoteInfo {
    double startOnset;
    double endOnset;
    List<PitchInfo> pitchList;
}
