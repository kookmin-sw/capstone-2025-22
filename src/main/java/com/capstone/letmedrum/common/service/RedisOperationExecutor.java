package com.capstone.letmedrum.common.service;

import lombok.extern.slf4j.Slf4j;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.data.redis.core.SessionCallback;
import org.springframework.stereotype.Component;

@Slf4j
@Component
public class RedisOperationExecutor {
    /**
     * run redis operation without exception
     * @param runnable redis operation
     * @return 1 if operation success
    * */
    public boolean execute(Runnable runnable) {
        try {
            runnable.run();
            return true;
        }catch (Exception e) {
            log.error(e.getMessage());
            return false;
        }
    }
}
