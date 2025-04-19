# update service
echo "$1"
docker build -t "$1":test -f ./"$1"/Dockerfile . && \
docker tag "$1":test louie8821/"$1":test && \
docker push louie8821/"$1":test