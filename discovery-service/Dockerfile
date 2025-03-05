# Build Stage
FROM openjdk:17-jdk-slim AS builder
LABEL authors="kym8821"
ARG MODULE_NAME=discovery-service
ARG COMMON_MODULE=common
WORKDIR /app
# copy from root module
COPY gradlew /app/
COPY gradle /app/gradle/
COPY build.gradle /app/
COPY docker-settings.gradle /app/settings.gradle
# copy from sub module
COPY /${MODULE_NAME} /app/module
# copy from common module
COPY /${COMMON_MODULE} /app/common
# ensure execute gradlew
RUN chmod +x gradlew
# build project
RUN ./gradlew build -x test --stacktrace
# copy spring boot jar on root dir
RUN cp module/build/libs/*.jar /app.jar
# Runtime Stage
FROM openjdk:17-jdk-slim
WORKDIR /app
# 빌드된 JAR 파일 복사 (JAR 파일이 여러 개일 경우 하나만 선택)
COPY --from=builder ./app.jar app.jar
# 컨테이너 실행 명령
ENTRYPOINT ["java", "-jar", "app.jar"]
