#/bin/bash
# init variable
score=0
success_commands=()
function update_score {
  score=$((score+16))
  success_commands+=("$1 update success\n")
}
# login to docker
docker login
# update auth-service
if sh ./common_updater.sh auth-service; then
  update_score auth-service
fi
# update user-service
if sh ./common_updater.sh user-service; then
  update_score user-service
fi
# update verification-service
if sh ./common_updater.sh verification-service; then
  update_score verification-service
fi
# update gateway-service
if sh ./common_updater.sh gateway-service; then
  update_score gateway-service
fi
# update config-server
if sh ./common_updater.sh config-server; then
  update_score config-server
fi
# update discovery-service
if sh ./common_updater.sh discovery-service; then
  update_score discovery-service
fi

for command in "${success_commands[@]}"; do
  echo -e "\033[32m$command\033[0m"
done

echo -e "\033[32mFinal Score: $score / 96 \033[0m"