package com.capstone.client;

import com.capstone.dto.musicXml.MeasureInfo;
import com.capstone.dto.score.FinalMeasureResult;
import com.capstone.enums.SuccessFlag;
import com.capstone.response.CustomResponseDto;
import com.fasterxml.jackson.databind.ObjectMapper;
import okhttp3.mockwebserver.MockResponse;
import org.apache.http.entity.ContentType;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import okhttp3.mockwebserver.MockWebServer;

import com.capstone.dto.sheet.MusicServiceClientDto.SheetPracticeCreateRequest;
import org.springframework.web.reactive.function.client.WebClient;

import java.io.IOException;
import java.util.List;

class MusicClientServiceTest {

    private MusicClientService musicClientService;

    MockWebServer mockWebServer;

    @BeforeEach
    void setUp() throws IOException {
        this.mockWebServer = new MockWebServer();
        mockWebServer.start();
        String baseUrl = mockWebServer.url("/").toString();
        WebClient webClient = WebClient.builder()
                .baseUrl(baseUrl).build();
        musicClientService = new MusicClientService(webClient);
    }

    @Test
    void getMeasureInfo_success() throws Exception {
        // given
        int useSheetId = 1;
        String measureNumber = "1";
        // stub
        MeasureInfo measureInfo = MeasureInfo.builder()
                .measureNumber(measureNumber).build();
        CustomResponseDto<MeasureInfo> mockServerResponse = CustomResponseDto.<MeasureInfo>builder()
                .body(measureInfo).build();
        this.mockWebServer.enqueue(new MockResponse()
                .addHeader("Content-Type", ContentType.APPLICATION_JSON.getMimeType())
                .setBody(new ObjectMapper().writeValueAsString(mockServerResponse)));
        // when
        MeasureInfo res = musicClientService.getMeasureInfo(useSheetId, measureNumber).block();
        // then
        assert res!=null;
        assert res.getMeasureNumber().equals(measureNumber);
        assert this.mockWebServer.getRequestCount()==1;
    }

    @Test
    void saveMeasureScoreInfo() throws Exception {
        // given
        String measureNumber = "1";
        FinalMeasureResult finalMeasureResult = FinalMeasureResult.builder()
                .measureNumber(measureNumber)
                .build();
        List<FinalMeasureResult> finalMeasureResults = List.of(finalMeasureResult);
        SheetPracticeCreateRequest requestDto = SheetPracticeCreateRequest.builder()
                .finalMeasures(finalMeasureResults)
                .score(100.0).build();
        // stub
        CustomResponseDto<String> mockServerResponse = CustomResponseDto.<String>builder()
                .body(SuccessFlag.SUCCESS.getLabel()).build();
        this.mockWebServer.enqueue(new MockResponse()
                .addHeader("Content-Type", ContentType.APPLICATION_JSON.getMimeType())
                .setBody(new ObjectMapper().writeValueAsString(mockServerResponse)));
        // when
        boolean res = Boolean.TRUE.equals(musicClientService.saveMeasureScoreInfo(requestDto).block());
        // then
        assert res;
    }
}