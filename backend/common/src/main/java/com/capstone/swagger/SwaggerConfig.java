package com.capstone.swagger;

import com.capstone.constants.AuthConstants;
import io.swagger.v3.oas.models.Components;
import io.swagger.v3.oas.models.OpenAPI;
import io.swagger.v3.oas.models.info.Info;
import io.swagger.v3.oas.models.security.SecurityRequirement;
import io.swagger.v3.oas.models.security.SecurityScheme;
import io.swagger.v3.oas.models.servers.Server;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.util.List;

@Configuration
public class SwaggerConfig {
    @Value("${gateway.service-url}")
    private String gatewayUrl;
    @Bean
    public OpenAPI openAPI() {
        String name = "Bearer Token";
        SecurityRequirement securityRequirement = new SecurityRequirement().addList(name);
        Components components = new Components().addSecuritySchemes(name, new SecurityScheme()
                .type(SecurityScheme.Type.HTTP)
                .in(SecurityScheme.In.HEADER)
                .name(AuthConstants.ACCESS_TOKEN_HEADER_KEY)
                .scheme("bearer")
                .bearerFormat("JWT")
        );
        Server server = new Server();
        server.setUrl(gatewayUrl);
        return new OpenAPI()
                .servers(List.of(server))
                .components(new Components())
                .info(apiInfo())
                .addSecurityItem(securityRequirement)
                .components(components);
    }
    private Info apiInfo() {
        return new Info()
                .title("알려드럼 API 테스트")
                .description("간단한 설명")
                .version("1.0.0");
    }
}