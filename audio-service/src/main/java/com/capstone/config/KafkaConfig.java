package com.capstone.config;

import org.apache.kafka.clients.admin.AdminClientConfig;
import org.apache.kafka.clients.admin.NewTopic;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.kafka.config.TopicBuilder;
import org.springframework.kafka.core.KafkaAdmin;

import java.util.HashMap;
import java.util.Map;

@Configuration
public class KafkaConfig {
    public final String bootstrapServer;
    public KafkaConfig(@Value("${spring.kafka.bootstrap-servers}") String bootstrapServer){
        this.bootstrapServer = bootstrapServer;
    }
    @Bean
    public KafkaAdmin kafkaAdmin(){
        Map<String, Object> config = new HashMap<>();
        config.put(AdminClientConfig.BOOTSTRAP_SERVERS_CONFIG, bootstrapServer);
        return new KafkaAdmin(config);
    }
    @Bean
    public NewTopic audioTopic(){
        return TopicBuilder.name("audio").build();
    }
}
