package com.capstone.letmedrum.common.service;

import org.springframework.data.redis.core.RedisTemplate;

public interface RedisService {
    public int setValue(String key, String value, int expire);
    public String getValue(String key);
    public int deleteValue(String key);
}
