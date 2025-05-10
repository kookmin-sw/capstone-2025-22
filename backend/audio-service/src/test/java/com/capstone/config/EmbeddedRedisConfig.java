package com.capstone.config;

import jakarta.annotation.PostConstruct;
import jakarta.annotation.PreDestroy;
import org.springframework.context.annotation.Configuration;
import redis.embedded.RedisServer;

import java.io.IOException;
import java.net.ServerSocket;

@Configuration
public class EmbeddedRedisConfig {

    private RedisServer redisServer;

    public int getAvailablePort() {
        try (ServerSocket socket = new ServerSocket(0)) {
            socket.setReuseAddress(true);
            return socket.getLocalPort();
        } catch (IOException e) {
            throw new IllegalStateException("No available port", e);
        }
    }

    @PostConstruct
    public void startRedis() {
        try {
            int port = getAvailablePort();
            redisServer = new RedisServer(port);
            redisServer.start();
        }catch (Exception e){
            throw new RuntimeException("failed to start redis server");
        }
    }

    @PreDestroy
    public void stopRedis() {
        if (redisServer != null) {
            redisServer.stop();
        }
    }
}
