# update service
echo "$1" "$2"
docker build -t "$1":test -f ./"$1"/Dockerfile . && \
docker tag "$1":test louie8821/"$1":"$2" && \
docker push louie8821/"$1":"$2"