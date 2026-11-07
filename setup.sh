#!/bin/sh
# Первичная настройка: создаёт .env и запускает проект.
set -e
cd "$(dirname "$0")"
if [ ! -f .env ]; then
  cp .env.example .env
  printf "IMOU APP_ID: "; read APP_ID
  printf "IMOU APP_SECRET: "; read APP_SECRET
  printf "Дата-центр Imou (sg / fk / or, Enter = sg): "; read DC; DC=${DC:-sg}
  SECRET=$(head -c 16 /dev/urandom | od -An -tx1 | tr -d ' \n')
  PASS=$(head -c 12 /dev/urandom | od -An -tx1 | tr -d ' \n')
  sed -i.bak \
    -e "s|^IMOU_APP_ID=.*|IMOU_APP_ID=$APP_ID|" \
    -e "s|^IMOU_APP_SECRET=.*|IMOU_APP_SECRET=$APP_SECRET|" \
    -e "s|^IMOU_API_HOST=.*|IMOU_API_HOST=https://openapi-$DC.easy4ip.com|" \
    -e "s|^WEBHOOK_SECRET=.*|WEBHOOK_SECRET=$SECRET|" \
    -e "s|^ADMIN_PASSWORD=.*|ADMIN_PASSWORD=$PASS|" .env
  rm -f .env.bak
  echo ".env создан. Пароль админки: $PASS (логин admin)"
fi
IP=$(grep '^PUBLIC_IP=' .env | cut -d= -f2)
[ -z "$IP" ] && IP=$(curl -fsS https://api.ipify.org || true)
DOM=$(grep '^DOMAIN=' .env | cut -d= -f2)
[ -z "$DOM" ] && DOM="$(echo "$IP" | tr '.' '-').sslip.io"
docker compose up -d --build
echo
echo "Готово. Админка: https://$DOM/"
echo "Если сертификат не выдаётся — проверьте, что порты 80 и 443 открыты (ufw allow 80,443/tcp)."
