package com.capstone.config;

import org.springframework.boot.test.context.TestConfiguration;
import org.springframework.context.annotation.Bean;
import org.springframework.kafka.test.EmbeddedKafkaBroker;
import org.springframework.kafka.test.EmbeddedKafkaZKBroker;

@TestConfiguration
public class EmbeddedKafkaConfig {
    @Bean
    public EmbeddedKafkaBroker embeddedKafkaBroker() {
        return new EmbeddedKafkaZKBroker(1, true, 1)
                .brokerProperty("controlled.shutdown", "true");
    }
}
