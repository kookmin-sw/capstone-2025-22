# Build Stage
FROM openjdk:17-jdk-slim AS builder
LABEL authors="kym8821"
WORKDIR /app
# copy from root module
COPY gradlew /app/
COPY gradle /app/gradle/
COPY build.gradle /app/
COPY docker-settings.gradle /app/settings.gradle
# copy from sub module
ARG MODULE_NAME=audio-service
COPY /${MODULE_NAME} /app/module
# copy from common module
ARG COMMON_MODULE1="common"
ARG COMMON_MODULE2="common-auth"
ARG COMMON_MODULE3="common-redis"
COPY /${COMMON_MODULE1} /app/${COMMON_MODULE1}
COPY /${COMMON_MODULE2} /app/${COMMON_MODULE2}
COPY /${COMMON_MODULE3} /app/${COMMON_MODULE3}
# copy config module
ARG CONFIG_MODULE="config-backend"
COPY /${CONFIG_MODULE} /app/${CONFIG_MODULE}
# ensure execute gradlew
RUN chmod +x gradlew
# build project
RUN ./gradlew build --build-cache --info
# copy spring boot jar on root dir
RUN ls module/build/libs
RUN cp module/build/libs/module-*.jar /app.jar

# Runtime Stage
FROM openjdk:17-jdk-slim
WORKDIR /app
# 빌드된 JAR 파일 복사 (JAR 파일이 여러 개일 경우 하나만 선택)
COPY --from=builder ./app.jar app.jar
# curl 설치
RUN apt-get update && apt-get install -y curl
# 컨테이너 실행 명령
ENTRYPOINT ["java", "-XX:-UseContainerSupport", "-jar", "app.jar"]
