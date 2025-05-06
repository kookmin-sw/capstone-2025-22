package com.capstone.sheet.constants;

public class MusicXmlTags {
    public static final String PART = "part";
    public static final String MEASURE = "measure";
    public static final String CHORD = "chord";
    public static final String NOTE = "note";
    public static final String NOTE_HEAD = "note-head";
    public static final String TYPE = "type";
    public static final String ATTRIBUTES = "attributes";
    public static final String PITCH = "pitch";
    public static final String DISPLAY_INFO = "display-info";
    public static final String DIVISIONS = "divisions";
    public static final String DURATION = "duration";
    public static final String TIME = "time";
    public static final String BEATS = "beats";
    public static final String BEAT_TYPE = "beat-type";

    public static class Pitch {
        public static final String REST = "rest";
        public static final String UNPITCHED = "unpitched";
        public static final String PITCHED = "pitched";
    }

    public static class DisplayInfo {
        public static final String DISPLAY_STEP = "display-step";
        public static final String DISPLAY_OCTAVE = "display-octave";
    }
}
