package com.capstone.service;

import com.capstone.constants.DrumInstrument;
import com.capstone.dto.ModelDto;
import com.capstone.dto.musicXml.MeasureInfo;
import com.capstone.dto.musicXml.NoteInfo;
import com.capstone.dto.musicXml.PitchInfo;
import com.capstone.dto.score.OnsetMatchResult;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class PracticeResultResolverTest {

    private final PracticeResultResolver resultResolver = new PracticeResultResolver();

    List<OnsetMatchResult> onsetMatchResults;
    List<String> userOnset;
    List<String[]> drumPredictList;
    MeasureInfo measureInfoMustBe100;

    public MeasureInfo getTestMeasureInfo(List<String[]> instrumentTypesList){
        double currentOnset = 0.0;
        List<NoteInfo> noteInfoList = new ArrayList<>();
        for(String[] instrumentTypes : instrumentTypesList){
            List<PitchInfo> pitchInfoList = new ArrayList<>();
            for(String instrumentType : instrumentTypes){
                pitchInfoList.add(PitchInfo.builder()
                        .instrumentType(instrumentType)
                        .build());
            }
            noteInfoList.add(NoteInfo.builder()
                    .pitchList(pitchInfoList)
                    .startOnset(currentOnset)
                    .endOnset(currentOnset+=1)
                    .build());
        }
        return MeasureInfo.builder()
                .noteList(noteInfoList)
                .build();
    }

    @BeforeEach
    void setUp() {
        OnsetMatchResult onsetMatchResultCorrect5 = OnsetMatchResult.builder()
                .userOnset(List.of(1.0, 2.0, 3.0, 4.0, 5.0))
                .answerOnset(List.of(1.0, 2.0, 3.0, 4.0, 5.0))
                .matchedUserOnsetIndices(new int[]{0, 1, 2, 3, 4})
                .answerOnsetPlayed(new boolean[]{true, true, true, true, true})
                .build();
        OnsetMatchResult onsetMatchResultCorrect4 = OnsetMatchResult.builder()
                .userOnset(List.of(1.0, 2.0, 3.0, 4.0, 5.0))
                .answerOnset(List.of(1.0, 2.0, 3.0, 4.5, 5.0))
                .matchedUserOnsetIndices(new int[]{0, 1, 2, -1, 4})
                .answerOnsetPlayed(new boolean[]{true, true, true, true, true})
                .build();
        OnsetMatchResult onsetMatchResultShort = OnsetMatchResult.builder()
                .userOnset(List.of(1.0, 2.0, 3.0, 4.0))
                .answerOnset(List.of(1.0, 2.0, 3.0, 4.5, 5.0))
                .matchedUserOnsetIndices(new int[]{0, 1, 2, -1})
                .answerOnsetPlayed(new boolean[]{true, true, true, false, false})
                .build();
        OnsetMatchResult onsetMatchResultLong = OnsetMatchResult.builder()
                .userOnset(List.of(1.0, 2.0, 3.0, 5.0, 6.0, 7.0))
                .answerOnset(List.of(1.0, 2.0, 3.0, 4.5, 5.0))
                .matchedUserOnsetIndices(new int[]{0, 1, 2, 4, -1, -1})
                .answerOnsetPlayed(new boolean[]{true, true, true, false, true})
                .build();
        onsetMatchResults = List.of(onsetMatchResultCorrect5, onsetMatchResultCorrect4, onsetMatchResultShort, onsetMatchResultLong);
        userOnset = List.of("0.0", "1.0", "2.0", "3.0", "4.0");
        drumPredictList = List.of(
                new String[]{DrumInstrument.SNARE, DrumInstrument.TOM},
                new String[]{DrumInstrument.TOM, DrumInstrument.SNARE},
                new String[]{DrumInstrument.TOM},
                new String[]{DrumInstrument.SNARE},
                new String[]{DrumInstrument.KICK}
        );
        measureInfoMustBe100 = getTestMeasureInfo(drumPredictList);
    }

    @Test
    void matchOnset_success(){
        // given
        List<String> shortUserOnset = List.of("0.0", "1.0", "2.0", "3.0");
        List<String> longUserOnset = List.of("0.0", "1.0", "2.0", "3.0", "4.0", "5.0");
        ModelDto.OnsetResponseDto commonOnsetResponseDto = ModelDto.OnsetResponseDto.builder()
                .onsets(userOnset)
                .build();
        ModelDto.OnsetResponseDto shortOnsetResponseDto = ModelDto.OnsetResponseDto.builder()
                .onsets(shortUserOnset)
                .build();
        ModelDto.OnsetResponseDto longOnsetResponseDto = ModelDto.OnsetResponseDto.builder()
                .onsets(longUserOnset)
                .build();
        MeasureInfo measureInfo = this.getTestMeasureInfo(drumPredictList);
        // when
        OnsetMatchResult shortOnsetMatchResult = resultResolver.matchOnset(shortOnsetResponseDto, measureInfo, 1);
        OnsetMatchResult commonOnsetMatchResult = resultResolver.matchOnset(commonOnsetResponseDto, measureInfo, 1);
        OnsetMatchResult longOnsetMatchResult = resultResolver.matchOnset(longOnsetResponseDto, measureInfo, 1);
        // then
        assert Arrays.equals(shortOnsetMatchResult.getAnswerOnsetPlayed(), new boolean[]{true, true, true, true, false});
        assert Arrays.equals(commonOnsetMatchResult.getAnswerOnsetPlayed(), new boolean[]{true, true, true, true, true});
        assert Arrays.equals(longOnsetMatchResult.getAnswerOnsetPlayed(), new boolean[]{true, true, true, true, true});

    }

    @Test
    void calculateScore_success() {
        // given
        List<Boolean> mustBe100 = List.of(true, true, true, true, true, true);
        List<Boolean> mustBe50 = List.of(true, true, true, false, false, false);
        List<Boolean> mustBe0 = List.of(false, false, false, false, false, false);
        // when
        double scoreMust70 = resultResolver.calculateScore(mustBe100, mustBe0);
        double scoreMust85 = resultResolver.calculateScore(mustBe100, mustBe50);
        double scoreMust100 = resultResolver.calculateScore(mustBe100, mustBe100);
        double scoreMust0 = resultResolver.calculateScore(mustBe0, mustBe0);
        // then
        assertEquals(100.0, scoreMust100);
        assertEquals(85.0, scoreMust85);
        assertEquals(70.0, scoreMust70);
        assertEquals(0, scoreMust0);
    }

    @Test
    void calculateFinalMatchingResult_success(){
        // given
        OnsetMatchResult onsetMatchResultCorrect5 = onsetMatchResults.get(0);
        OnsetMatchResult onsetMatchResultCorrect4 = onsetMatchResults.get(1);
        OnsetMatchResult onsetMatchResultShort = onsetMatchResults.get(2);
        OnsetMatchResult onsetMatchResultLong = onsetMatchResults.get(3);
        List<String[]> drumPredictList = this.drumPredictList;
        MeasureInfo measureInfo = this.measureInfoMustBe100;
        // when
        List<Boolean> mustBeTTTTT = resultResolver.calculateFinalMatchingResult(onsetMatchResultCorrect5, drumPredictList, measureInfo);
        List<Boolean> mustBeTTTFT = resultResolver.calculateFinalMatchingResult(onsetMatchResultCorrect4, drumPredictList, measureInfo);
        List<Boolean> mustBeTTTFFShort = resultResolver.calculateFinalMatchingResult(onsetMatchResultShort, drumPredictList, measureInfo);
        List<Boolean> mustBeTTTFFLong = resultResolver.calculateFinalMatchingResult(onsetMatchResultLong, drumPredictList, measureInfo);
        // then
        assert mustBeTTTTT.equals(List.of(true, true, true, true, true));
        assert mustBeTTTFT.equals(List.of(true, true, true, false, true));
        assert mustBeTTTFFShort.equals(List.of(true, true, true, false, false));
        assert mustBeTTTFFShort.size() == onsetMatchResultShort.getAnswerOnset().size();
        assert mustBeTTTFFLong.equals(List.of(true, true, true, false, false));
        assert mustBeTTTFFLong.size() == onsetMatchResultLong.getAnswerOnset().size();
    }

    @Test
    void calculateBeatMatchingResult_success(){
        // given
        OnsetMatchResult onsetMatchResultCorrect5 = onsetMatchResults.get(0);
        OnsetMatchResult onsetMatchResultCorrect4 = onsetMatchResults.get(1);
        OnsetMatchResult onsetMatchResultShort = onsetMatchResults.get(2);
        OnsetMatchResult onsetMatchResultLong = onsetMatchResults.get(3);
        // when
        List<Boolean> mustBeTTTTT = resultResolver.calculateBeatMatchingResult(onsetMatchResultCorrect5);
        List<Boolean> mustBeTTTFT = resultResolver.calculateBeatMatchingResult(onsetMatchResultCorrect4);
        List<Boolean> mustBeTTTFFShort = resultResolver.calculateBeatMatchingResult(onsetMatchResultShort);
        List<Boolean> mustBeTTTFTLong = resultResolver.calculateBeatMatchingResult(onsetMatchResultLong);
        // then
        assert mustBeTTTTT.equals(List.of(true, true, true, true, true));
        assert mustBeTTTFT.equals(List.of(true, true, true, false, true));
        assert mustBeTTTFFShort.equals(List.of(true, true, true, false, false));
        assert mustBeTTTFFShort.size() == onsetMatchResultShort.getAnswerOnset().size();
        assert mustBeTTTFTLong.equals(List.of(true, true, true, false, true));
        assert mustBeTTTFTLong.size() == onsetMatchResultLong.getAnswerOnset().size();
    }
}