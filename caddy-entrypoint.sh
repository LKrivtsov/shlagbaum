#!/bin/sh
# Определяем адрес сайта: свой DOMAIN или <ip>.sslip.io
if [ -z "$PUBLIC_IP" ]; then
  for u in http://api.ipify.org http://ifconfig.me/ip http://icanhazip.com; do
    PUBLIC_IP=$(wget -qO- -T 5 "$u" 2>/dev/null | tr -d '[:space:]')
    [ -n "$PUBLIC_IP" ] && break
  done
fi
if [ -z "$PUBLIC_IP" ]; then
  echo "Не удалось определить публичный IP. Укажите PUBLIC_IP в .env" >&2
  exit 1
fi
if [ -z "$DOMAIN" ]; then
  DOMAIN="$(echo "$PUBLIC_IP" | tr '.' '-').sslip.io"
fi
export DOMAIN PUBLIC_IP
echo "Сайт: https://$DOMAIN/"
exec caddy run --config /etc/caddy/Caddyfile --adapter caddyfile
