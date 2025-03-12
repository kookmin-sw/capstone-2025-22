package com.capstone.config;

import org.springframework.kafka.core.ConsumerFactory;

public class KafkaConsumerConfig {
    private final KafkaConfig kafkaConfig;
    public KafkaConsumerConfig(KafkaConfig kafkaConfig){
        this.kafkaConfig = kafkaConfig;
    }
}
