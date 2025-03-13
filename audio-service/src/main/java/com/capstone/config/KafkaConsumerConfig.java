package com.capstone.config;

import com.capstone.dto.AudioMessageDto;
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
    public JsonDeserializer<AudioMessageDto> getAudioMessageDeserializer() {
        JsonDeserializer<AudioMessageDto> deserializer = new JsonDeserializer<>(AudioMessageDto.class, new ObjectMapper());
        deserializer.addTrustedPackages("com.capstone.dto");
        deserializer.setUseTypeMapperForKey(true);
        return deserializer;
    }
    @Bean
    public ConsumerFactory<String, AudioMessageDto> consumerFactory(){
        Map<String, Object> config = new HashMap<>();
        config.put(ConsumerConfig.BOOTSTRAP_SERVERS_CONFIG, kafkaConfig.bootstrapServer);
        config.put(ConsumerConfig.KEY_DESERIALIZER_CLASS_CONFIG, StringDeserializer.class);
        config.put(ConsumerConfig.VALUE_DESERIALIZER_CLASS_CONFIG, JsonDeserializer.class);
        config.put(ConsumerConfig.GROUP_ID_CONFIG, groupId);
        return new DefaultKafkaConsumerFactory<>(config, new StringDeserializer(), getAudioMessageDeserializer());
    }
    @Bean
    public KafkaListenerContainerFactory<?> kafkaListenerContainerFactory(){
        ConcurrentKafkaListenerContainerFactory<String, AudioMessageDto> factory = new ConcurrentKafkaListenerContainerFactory<>();
        factory.setConsumerFactory(consumerFactory());
        return factory;
    }
}
