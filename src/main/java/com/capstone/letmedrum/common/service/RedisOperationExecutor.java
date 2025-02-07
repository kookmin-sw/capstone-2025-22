package com.capstone.letmedrum.common.service;

import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

@Slf4j
@Component
public class RedisOperationExecutor {
    /**
     * run redis operation more safe
     * @param runnable redis operation
     * @return 1 if operation success
    * */
    public int execute(Runnable runnable) {
        try {
            runnable.run();
            return 1;
        }catch (Exception e) {
            log.error(e.getMessage());
            return 0;
        }
    }
}
