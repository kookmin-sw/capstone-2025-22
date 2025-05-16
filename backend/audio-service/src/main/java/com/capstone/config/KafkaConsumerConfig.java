package com.capstone.config;

import com.capstone.dto.AudioMessageDto;
import com.capstone.dto.PatternMessageDto;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.kafka.clients.consumer.ConsumerConfig;
import org.apache.kafka.common.serialization.StringDeserializer;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.kafka.config.ConcurrentKafkaListenerContainerFactory;
import org.springframework.kafka.config.KafkaListenerContainerFactory;
import org.springframework.kafka.core.ConsumerFactory;
import org.springframework.kafka.core.DefaultKafkaConsumerFactory;
import org.springframework.kafka.support.serializer.JsonDeserializer;

import java.util.HashMap;
import java.util.Map;

@Configuration
public class KafkaConsumerConfig {
    private final KafkaConfig kafkaConfig;
    private final String groupId;

    public KafkaConsumerConfig(KafkaConfig kafkaConfig, @Value("${spring.kafka.consumer.group-id}") String groupId){
        this.groupId = groupId;
        this.kafkaConfig = kafkaConfig;
    }

    public <T> JsonDeserializer<T> getJsonDeserializer(Class<T> clazz) {
        JsonDeserializer<T> deserializer = new JsonDeserializer<>(clazz, new ObjectMapper());
        deserializer.addTrustedPackages("com.capstone.dto");
        deserializer.setUseTypeMapperForKey(true);
        return deserializer;
    }

    public <T> ConsumerFactory<String, T> consumerFactory(Class<T> clazz){
        Map<String, Object> config = new HashMap<>();
        config.put(ConsumerConfig.BOOTSTRAP_SERVERS_CONFIG, kafkaConfig.bootstrapServer);
        config.put(ConsumerConfig.KEY_DESERIALIZER_CLASS_CONFIG, StringDeserializer.class);
        config.put(ConsumerConfig.VALUE_DESERIALIZER_CLASS_CONFIG, JsonDeserializer.class);
        config.put(ConsumerConfig.GROUP_ID_CONFIG, groupId);
        return new DefaultKafkaConsumerFactory<>(config, new StringDeserializer(), getJsonDeserializer(clazz));
    }

    public <T> ConcurrentKafkaListenerContainerFactory<String, T> createKafkaListenerFactory(Class<T> clazz){
        ConcurrentKafkaListenerContainerFactory<String, T> factory = new ConcurrentKafkaListenerContainerFactory<>();
        factory.setConsumerFactory(consumerFactory(clazz));
        return factory;
    }

    @Bean
    public ConcurrentKafkaListenerContainerFactory<String, AudioMessageDto> audioKafkaListenerFactory(){
        return createKafkaListenerFactory(AudioMessageDto.class);
    }

    @Bean
    public ConcurrentKafkaListenerContainerFactory<String, PatternMessageDto> patternKafkaListenerFactory(){
        return createKafkaListenerFactory(PatternMessageDto.class);
    }
}
