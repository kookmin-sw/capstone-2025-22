package com.capstone.config;

import com.capstone.dto.AudioMessageDto;
import com.capstone.dto.PatternMessageDto;
import org.apache.kafka.clients.producer.ProducerConfig;
import org.apache.kafka.common.serialization.StringSerializer;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.kafka.core.DefaultKafkaProducerFactory;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.kafka.core.ProducerFactory;
import org.springframework.kafka.support.serializer.JsonSerializer;

import java.util.HashMap;
import java.util.Map;

@Configuration
public class KafkaProducerConfig{
    private final KafkaConfig kafkaConfig;
    public KafkaProducerConfig(KafkaConfig kafkaConfig){
        this.kafkaConfig = kafkaConfig;
    }

    public <T> ProducerFactory<String, T> createProducerFactory(){
        Map<String, Object> config = new HashMap<>();
        config.put(ProducerConfig.BOOTSTRAP_SERVERS_CONFIG, kafkaConfig.bootstrapServer);
        config.put(ProducerConfig.KEY_SERIALIZER_CLASS_CONFIG, StringSerializer.class);
        config.put(ProducerConfig.VALUE_SERIALIZER_CLASS_CONFIG, JsonSerializer.class);
        return new DefaultKafkaProducerFactory<>(config);
    }

    public <T> KafkaTemplate<String, T> kafkaTemplate(){
        return new KafkaTemplate<>(createProducerFactory());
    }

    @Bean
    public KafkaTemplate<String, AudioMessageDto> audioKafkaTemplate(){
        return new KafkaTemplate<>(createProducerFactory());
    }

    @Bean
    public KafkaTemplate<String, PatternMessageDto> patternKafkaTemplate(){
        return new KafkaTemplate<>(createProducerFactory());
    }
}
