package com.capstone.service;

import com.capstone.dto.score.FinalMeasureResult;
import com.capstone.redis.RedisSingleDataServiceImpl;
import lombok.RequiredArgsConstructor;
import org.springframework.data.redis.connection.ReactiveRedisConnection;
import org.springframework.data.redis.core.ReactiveRedisTemplate;
import org.springframework.data.redis.core.ScanOptions;
import org.springframework.stereotype.Component;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import java.util.List;

@Component
@RequiredArgsConstructor
public class MeasureScoreManager {
    private final RedisSingleDataServiceImpl redisService;
    private final ReactiveRedisTemplate<String, String> reactiveRedisTemplate;

    public String getMeasureScoreKey(String identifier, String measureNumber){
        return String.format("practice-%s-%s", identifier, measureNumber);
    }

    public Mono<String> getMeasureScore(String identifier, String measureNumber){
        String key = getMeasureScoreKey(identifier, measureNumber);
        return redisService.getValue(key);
    }

    public Mono<Boolean> saveMeasureScore(String identifier, String measureNumber, FinalMeasureResult result){
        String key = getMeasureScoreKey(identifier, measureNumber);
        return redisService.setValue(key, result.toString(), 3600);
    }

    public Mono<List<String>> getAllMeasureScores(String identifier){
        return reactiveRedisTemplate.keys("practice-" + identifier + "-*")
                .flatMap(redisService::getValue)
                .collectList();
    }

    public List<Long> deleteAllMeasureScores(String identifier){
        ScanOptions scanOptions = ScanOptions.scanOptions().match("practice-" + identifier + "-*").build();
        ReactiveRedisConnection connection = reactiveRedisTemplate.getConnectionFactory().getReactiveConnection();
        return connection.keyCommands().scan(scanOptions)
                .map(byteBuffer -> {
                    byte[] bytes = new byte[byteBuffer.remaining()];
                    byteBuffer.get(bytes);
                    return new String(bytes);
                })
                .collectList()
                .flatMapMany(keys ->
                        reactiveRedisTemplate.delete(Flux.fromIterable(keys))
                ).collectList().block();
    }
}
