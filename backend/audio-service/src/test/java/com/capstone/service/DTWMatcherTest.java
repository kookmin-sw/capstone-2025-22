package com.capstone.service;

import com.capstone.dto.score.OnsetMatchResult;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.List;

class DTWMatcherTest {

    @Test
    void match() {
        // given
        List<Double> userOnset1 = List.of(1.0, 2.0, 3.0, 4.0, 5.0);
        List<Double> userOnset2 = List.of(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
        List<Double> userOnset3 = List.of(1.0, 2.0, 4.0, 5.0);
        List<Double> userOnset4 = List.of(1.0, 2.0, 3.9, 4.0, 5.0);
        List<Double> answerOnset = List.of(1.0, 2.0, 3.0, 4.0, 5.0);
        // when
        OnsetMatchResult result1 = DTWMatcher.match(userOnset1, answerOnset, 0.0, 1);
        OnsetMatchResult result2 = DTWMatcher.match(userOnset2, answerOnset, 0.0, 1);
        OnsetMatchResult result3 = DTWMatcher.match(userOnset3, answerOnset, 0.0, 1);
        OnsetMatchResult result4 = DTWMatcher.match(userOnset4, answerOnset, 0.0, 1);
        // then
        assert Arrays.equals(result1.getAnswerOnsetPlayed(), new boolean[]{true, true, true, true, true});
        assert Arrays.equals(result1.getMatchedUserOnsetIndices(), new int[]{0, 1, 2, 3, 4});
        assert Arrays.equals(result2.getAnswerOnsetPlayed(), new boolean[]{true, true, true, true, true});
        assert Arrays.equals(result2.getMatchedUserOnsetIndices(), new int[]{0, 1, 2, 3, 4, -1});
        assert Arrays.equals(result3.getAnswerOnsetPlayed(), new boolean[]{true, true, false, true, true});
        assert Arrays.equals(result3.getMatchedUserOnsetIndices(), new int[]{0, 1, 3, 4});
        assert Arrays.equals(result4.getAnswerOnsetPlayed(), new boolean[]{true, true, false, true, true});
        assert Arrays.equals(result4.getMatchedUserOnsetIndices(), new int[]{0, 1, -1, 3, 4});
    }
}