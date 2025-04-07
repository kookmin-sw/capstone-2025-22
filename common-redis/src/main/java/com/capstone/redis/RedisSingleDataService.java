package com.capstone.redis;

import reactor.core.publisher.Mono;

public interface RedisSingleDataService {
    Mono<Boolean> setValue(String key, String value, int expire);
    Mono<String> getValue(String key);
    Mono<Boolean> deleteValue(String key);
}
