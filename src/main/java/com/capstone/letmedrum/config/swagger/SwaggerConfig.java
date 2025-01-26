package com.capstone.letmedrum.config.swagger;

import com.capstone.letmedrum.config.security.JwtUtils;
import io.swagger.v3.oas.models.Components;
import io.swagger.v3.oas.models.OpenAPI;
import io.swagger.v3.oas.models.info.Info;
import io.swagger.v3.oas.models.security.SecurityRequirement;
import io.swagger.v3.oas.models.security.SecurityScheme;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class SwaggerConfig {
    @Bean
    public OpenAPI openAPI() {
        String name = "Bearer Token";
        SecurityRequirement securityRequirement = new SecurityRequirement().addList(name);
        Components components = new Components().addSecuritySchemes(name, new SecurityScheme()
                .type(SecurityScheme.Type.HTTP)
                .in(SecurityScheme.In.HEADER)
                .name(JwtUtils.ACCESS_TOKEN_HEADER_KEY)
                .scheme("bearer")
                .bearerFormat("JWT")
        );
        return new OpenAPI()
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