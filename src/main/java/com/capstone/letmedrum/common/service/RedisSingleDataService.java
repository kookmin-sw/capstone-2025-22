package com.capstone.letmedrum.common.service;

import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.stereotype.Service;

import java.util.concurrent.TimeUnit;

@Service
public class RedisSingleDataService implements RedisService {
    private final RedisTemplate<String, Object> redisTemplate;
    private final RedisOperationExecutor redisOperationExecutor;
    /**
     * constructor for DI
     * @param redisTemplate RedisTemplate Bean on RedisConfig
     * @param redisOperationExecutor RedisOperationExecutor Bean
    * */
    public RedisSingleDataService(RedisTemplate<String, Object> redisTemplate, RedisOperationExecutor redisOperationExecutor) {
        this.redisTemplate = redisTemplate;
        this.redisOperationExecutor = redisOperationExecutor;
    }
    /**
     * set single value on storage
     * @param key data's key
     * @param value data's value
     * @param expSeconds Expiration time in seconds
     * @return 1 if success
    * */
    @Override
    public int setValue(String key, String value, int expSeconds) {
        return redisOperationExecutor.execute(() -> redisTemplate.opsForValue().set(key, value, expSeconds, TimeUnit.SECONDS));
    }
    /**
     * get single value from storage
     * @param key key of data trying to find
     * @return data if exists else return blank string ("")
    * */
    @Override
    public String getValue(String key) {
        Object value = redisTemplate.opsForValue().get(key);
        return value == null ? "" : value.toString();
    }
    /**
     * delete value with key
     * @param key key of data trying to delete
     * @return 1 if success
    * */
    @Override
    public int deleteValue(String key) {
        return redisOperationExecutor.execute(() -> redisTemplate.delete(key));
    }

}
