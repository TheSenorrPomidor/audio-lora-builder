🧠 Audio Lora Builder — Локальный ИИ на основе телефонных разговоров

Этот проект позволяет собрать датасет из твоих .m4a‑звонков для обучения LoRA‑модели под твой стиль речи.Работает внутри WSL2 (Ubuntu) полностью офлайн.

🚀 Установка (через PowerShell)

📌 Перед запуском:
Если при старте скрипта появляется ошибка вида:

выполнение сценариев отключено в этой системе
или
файл не имеет цифровой подписи

выполни в PowerShell (от имени администратора):

Set-ExecutionPolicy Bypass -Scope Process

🔒 Это временно разрешит запуск любых .ps1-файлов в этой сессии (до закрытия PowerShell).

📎 ⚠️ Важно: стандартные консоли Windows (cmd.exe, PowerShell) не отображают русские буквы и emoji корректно. Если вместо текста видна каша типа РІРµСЂСЃРёСЏ,

✅ Работай через Windows Terminal Portable — он отображает всё корректно и запускается без ограничений.

Скачай:

install_audio_lora.ps1

bootstrap_audio_lora.sh

process_audio.py

README.md

Помести все файлы в одну папку, например D:\AI\audio-lora\

Создай там же подпапку audio_src\ и положи туда свои .m4a, .mp3, .aac файлы. Поддерживается как вложенная структура, так и плоские файлы.

Открой PowerShell от имени администратора

Выполни:

Set-ExecutionPolicy RemoteSigned -Scope Process
cd D:\AI\audio-lora
.\install_audio_lora.ps1

📌 Скрипт автоматически:

скачивает Ubuntu 22.04 (jammy), если файл ещё не загружен,

устанавливает WSL2-дистрибутив прямо в папку скрипта,

определяет путь к bootstrap_audio_lora.sh,

копирует все аудиофайлы из audio_src/ внутрь WSL-папки input/

📁 Структура проекта в Windows (пример)

📂 D:\AI\audio-lora
├── install_audio_lora.ps1          ← главный PowerShell-установщик
├── bootstrap_audio_lora.sh         ← bash-скрипт для настройки Ubuntu
├── process_audio.py                ← скрипт обработки аудио
├── README.md                       ← инструкция
├── ubuntu_rootfs.tar.gz            ← скачивается автоматически
├── 📁 audio_src\
│   ├── 2023\05\17\+7916...\*.m4a ← вложенная структура (поддерживается)
│   ├── file001.mp3                 ← и плоские файлы тоже

📦 Куда устанавливается WSL-дистрибутив?

WSL-дистрибутив audio-lora будет установлен в ту же папку, где находится install_audio_lora.ps1, например:

D:\AI\audio-lora\audio-lora\

Туда попадёт вся Ubuntu: системные файлы, данные, кэшированные модели и сам проект. Убедись, что на диске достаточно места (рекомендуется 5–10 ГБ).

🐙 Что делает скрипт внутри Ubuntu (bootstrap_audio_lora.sh)

Обновляет пакеты в системе

Устанавливает Python 3, pip, ffmpeg, git, wget

Устанавливает Python-библиотеки: Whisper, WhisperX, pyannote.audio

Создаёт структуру проекта:

~/audio-lora-builder/
├── input/        ← копии из Windows/audio_src/
├── output/       ← сюда попадают результаты (.jsonl)
├── logs/         ← логи обработки
├── models/       ← кэш моделей whisper и pyannote
├── run.sh        ← запуск `process_audio.py`
├── process_audio.py ← скрипт обработки аудиофайлов

🔧 Запуск

После установки:

cd ~/audio-lora-builder
./run.sh

Или сразу из Windows (PowerShell):

wsl -d audio-lora -- ~/audio-lora-builder/run.sh

Это выполнит process_audio.py, который:

обходит input/ рекурсивно

находит .m4a, .mp3, .aac

выполняет распознавание речи через Whisper

сохраняет сегменты в dialogue.jsonl

🧠 Как включить модель large-v3 (точнее, но тяжелее)

Открой файл process_audio.py и найди строку:

model_size = "base"

Замени её на:

model_size = "large-v3"

Это увеличит точность распознавания, особенно на русском, но потребует больше памяти (видеокарта с 10+ ГБ VRAM или RAM-режим).

📆 Зависимости

Python 3.10+

ffmpeg

CUDA (опционально)

whisper / whisperx / pyannote.audio

⚡ Быстрая инструкция (tl;dr)

🟢 Установка:

Получить токен

🔐 Активация доступа к моделям pyannote.audio
Для загрузки моделей диаризации pyannote.audio требуется:

1. 📋 Получить токен Hugging Face
Зарегистрируйтесь или войдите на https://huggingface.co

Перейдите в настройки токенов

Нажмите "New token"

Укажите любое имя, выберите Scope: "Read", нажмите "Generate token"

Скопируйте полученный токен

2. 🔑 Авторизоваться в WSL
В терминале Linux (WSL) выполните:

bash
Копировать
Редактировать
huggingface-cli login
Вставьте токен. На вопрос Add token as git credential? — введите n.

3. ✅ Подтвердить доступ к gated-моделям
Перейдите в браузере и нажмите "Access repository" для каждой модели:

🔗 pyannote/speaker-diarization-3.1 👉 https://huggingface.co/pyannote/speaker-diarization-3.1

🔗 pyannote/segmentation 👉 https://huggingface.co/pyannote/segmentation

🔗 pyannote/embedding 👉 https://huggingface.co/pyannote/embedding

🔗 НЕ АКТУАЛЬНО pyannote/voice-activity-detection 👉 https://huggingface.co/pyannote/voice-activity-detection

💡 Без этого загрузка моделей завершится ошибкой Could not download... model is gated.





.\install_audio_lora.ps1  # от имени администратора

📦 Результат:

скачивается rootfs Ubuntu

создаётся папка audio-lora/ с дистрибутивом WSL

появляется ~/audio-lora-builder с input/output и всем нужным

копируются файлы из audio_src/

▶️ Запуск обработки:

wsl -d audio-lora -- ~/audio-lora-builder/run.sh

❌ Если что-то пошло не так:

Симптом

Решение

audio-lora/ нет

Запусти .ps1 от имени администратора

input/ пустой

Убедись, что есть audio_src/ с файлами

Ошибка в Ubuntu

wsl -d audio-lora и проверь вручную

Хочешь удалить всё

wsl --unregister audio-lora + удалить папку