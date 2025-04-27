FROM node:17

WORKDIR /app
ENV PATH /app/node_modules/.bin:$PATH

RUN ["apt-get", "update"]
RUN ["apt-get", "install", "build-essential", "libcairo2-dev", "libpango1.0-dev", "libjpeg-dev", "libgif-dev", "librsvg2-dev", "-y"]

# install package
COPY ["package*.json", "./"]
RUN ["npm", "install", "--silent"]

COPY [".", "."]

# ENTRYPOINT ["tail", "-f", "/dev/null"]