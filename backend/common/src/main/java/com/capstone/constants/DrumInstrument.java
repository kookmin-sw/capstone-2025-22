package com.capstone.constants;

public class DrumInstrument {
    public static final String SNARE = "SD";
    public static final String HIGH_HAT = "HH";
    public static final String KICK = "KD";
    public static final String CYMBAL = "CY";
    public static final String TOM = "TT";

    public static String getDrumInstrumentType(String instrumentName){
        if(instrumentName.contains("Snare")) return SNARE;
        else if(instrumentName.contains("Hat")) return HIGH_HAT;
        else if(instrumentName.contains("Kick") || instrumentName.contains("Drum")) return KICK;
        else if(instrumentName.contains("Cymbal")) return CYMBAL;
        else if(instrumentName.contains("Tom")) return TOM;
        return null;
    }
}
