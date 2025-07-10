#!/bin/bash
set -e

echo "🔧 Проверка DNS..."
if ! grep -q "nameserver" /etc/resolv.conf || ! ping -c1 archive.ubuntu.com &>/dev/null; then
    echo "⚠️  DNS недоступен. Добавляем 8.8.8.8..."
    echo "nameserver 8.8.8.8" | sudo tee /etc/resolv.conf > /dev/null
fi

echo "📦 Обновляем пакеты..."
sudo apt update
sudo apt upgrade -y

echo "🧰 Устанавливаем зависимости..."
sudo apt install -y python3-pip ffmpeg

echo "✅ bootstrap_audio_lora.sh завершён."
