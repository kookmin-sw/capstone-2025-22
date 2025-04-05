package com.capstone.redis;

import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.context.annotation.Profile;
import org.springframework.data.redis.core.ReactiveRedisTemplate;
import org.springframework.stereotype.Service;
import reactor.core.publisher.Mono;


@Service
@Profile("redis")
public class RedisSingleDataServiceImpl implements RedisSingleDataService {
    private final ReactiveRedisTemplate<String, String> redisTemplate;
    /**
     * constructor for DI
     * @param redisTemplate RedisTemplate Bean on RedisConfig
    * */
    public RedisSingleDataServiceImpl(@Qualifier("reactiveRedisTemplate") ReactiveRedisTemplate<String, String> redisTemplate) {
        this.redisTemplate = redisTemplate;
    }
    /**
     * set single value on storage
     * @param key data's key
     * @param value data's value
     * @param expSeconds Expiration time in seconds
     * @return 1 if success
    * */
    @Override
    public Mono<Boolean> setValue(String key, String value, int expSeconds) {
        return redisTemplate.opsForValue().set(key, value);
    }
    /**
     * get single value from storage
     * @param key key of data trying to find
     * @return data if exists else return blank string ("")
    * */
    @Override
    public Mono<String> getValue(String key) {
        return redisTemplate.opsForValue().get(key)
                .onErrorResume(throwable -> Mono.empty());
    }
    /**
     * delete value with key
     * @param key key of data trying to delete
     * @return 1 if success
    * */
    @Override
    public Mono<Boolean> deleteValue(String key) {
        return redisTemplate.delete(key)
                .map(v -> true);
    }

}
