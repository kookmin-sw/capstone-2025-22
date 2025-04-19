package com.capstone.redis;

import lombok.extern.slf4j.Slf4j;
import org.springframework.context.annotation.Profile;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.data.redis.core.SessionCallback;
import org.springframework.stereotype.Component;
import reactor.core.publisher.Mono;

import java.util.function.Supplier;

@Slf4j
@Component
@Profile("redis")
public class RedisOperationExecutor {
    /**
     * run redis operation without exception
     * @param operation redis operation
     * @return 1 if operation success
    * */
//    public <T> Mono<Boolean> execute(Supplier<Mono<T>> operation) {
//        return Mono.just(() ->
//                operation.get()
//                        .doOnError(e -> log.error("Redis operation failed: {}", e.getMessage()))
//        );
//    }
}
