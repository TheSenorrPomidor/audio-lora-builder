#!/usr/bin/env python3
# Версия скрипта 1.5

import os
import shutil
import subprocess
from pathlib import Path

# === 0. Версия ===
print("🔢 Версия скрипта process_audio.py 1.5")

# === 1. Чтение конфигурации ===
ENV_FILE = "/root/audio-lora-builder/config/env.vars"
WIN_AUDIO_SRC = None

if os.path.exists(ENV_FILE):
    with open(ENV_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("WIN_AUDIO_SRC="):
                WIN_AUDIO_SRC = line.strip().split("=", 1)[1]
                break

# Если не найдено — просим пользователя ввести вручную
if not WIN_AUDIO_SRC or not os.path.exists(WIN_AUDIO_SRC):
    print("⚠️ Не удалось найти корректный путь до папки с аудиофайлами.")
    print("💡 Укажите путь вручную. Пример: /mnt/c/Users/you/audio_src")
    user_input = input("Введите путь до папки с аудиофайлами [/mnt/c/]: ").strip()
    if not user_input:
        user_input = "/mnt/c/"
    WIN_AUDIO_SRC = user_input.replace("\\", "/")

    # Сохраняем путь в конфигурационный файл
    Path("/root/audio-lora-builder/config").mkdir(parents=True, exist_ok=True)
    with open(ENV_FILE, "w", encoding="utf-8") as f:
        f.write(f"WIN_AUDIO_SRC={WIN_AUDIO_SRC}\n")

# === 2. Конвертация файлов ===
print("🎧 Начинаем поиск и конвертацию аудиофайлов...")

AUDIO_EXTENSIONS = [".m4a", ".mp3", ".aac"]
SRC = Path(WIN_AUDIO_SRC)
DST = Path("/root/audio-lora-builder/input/audio_src")

if DST.exists():
    shutil.rmtree(DST)
DST.mkdir(parents=True, exist_ok=True)

files = [f for f in SRC.rglob("*") if f.suffix.lower() in AUDIO_EXTENSIONS]
for idx, file in enumerate(files, 1):
    relative = file.relative_to(SRC)
    output = DST / relative.with_suffix(".wav")
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"🎛 ({idx}) {file} → {output}")
    subprocess.run([
        "ffmpeg", "-y", "-i", str(file),
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
        str(output)
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

print(f"✅ Всего обработано файлов: {len(files)}")









# === 3. Распознавание речи (Whisper large-v3, cuDNN-guarded GPU, таймер, очистка) ===
print("3. Распознавание речи (Whisper large-v3, cuDNN-guarded GPU, таймер, очистка) v11")

from faster_whisper import WhisperModel
import time
######################
#import huggingface_hub.file_download
#
#_original_download = huggingface_hub.file_download.hf_hub_download
#
#def traced_hf_download(*args, **kwargs):
#    print("📥 Huggingface загрузка:", args, kwargs)
#    return _original_download(*args, **kwargs)
#
#huggingface_hub.file_download.hf_hub_download = traced_hf_download
#######################


def has_cudnn():
    import glob
    return any(glob.glob("/usr/lib*/**/libcudnn*.so*", recursive=True))

def load_model():
    if has_cudnn():
        print("🧠 Обнаружен cuDNN — используем GPU (int8)...")
        return WhisperModel("large-v3", device="cuda", compute_type="int8", cpu_threads=4)
    else:
        print("🧠 cuDNN не найден — используем CPU (int8)...")
        return WhisperModel("large-v3", device="cpu", compute_type="int8", cpu_threads=4)

def format_hhmmss(seconds):
    mins, secs = divmod(int(seconds), 60)
    hrs, mins = divmod(mins, 60)
    return f"{hrs:02}:{mins:02}:{secs:02}"

# Очистка папки output
OUTPUT_DIR = Path("/root/audio-lora-builder/output")
if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print("🧹 Папка output/ очищена.")

wav_files = list(DST.rglob("*.wav"))

if not wav_files:
    print("⚠️ Нет .wav файлов для обработки в input/audio_src/")
else:
    print(f"🔍 Найдено {len(wav_files)} .wav файлов для распознавания.")

    start_time = time.time()
    model = load_model()

    summary_lines = []

    for idx, audio_path in enumerate(wav_files, 1):
        elapsed = format_hhmmss(time.time() - start_time)
        rel_path = audio_path.relative_to(DST)
        out_txt = OUTPUT_DIR / rel_path.with_suffix(".txt")
        out_txt.parent.mkdir(parents=True, exist_ok=True)

        print(f"📝 ({idx}/{len(wav_files)} {elapsed}) Распознаём: {rel_path}")

        try:
            segments, _ = model.transcribe(
                str(audio_path),
                language="ru",
                beam_size=5,
                vad_filter=True,
                vad_parameters={"threshold": 0.5}
            )
            segments = list(segments)

            transcript_lines = [f"=== Файл: {rel_path} ==="]
            for segment in segments:
                if segment.text.strip():
                    timestamp = f"[{segment.start:.3f} --> {segment.end:.3f}]"
                    transcript_lines.append(f"{timestamp}\n{segment.text.strip()}\n")

            with open(out_txt, "w", encoding="utf-8") as f_out:
                f_out.write("\n".join(transcript_lines))

            summary_lines.extend(transcript_lines)
            summary_lines.append("")

        except Exception as e:
            print(f"❌ Ошибка при обработке {rel_path}: {e}")
            continue

    total_time = time.time() - start_time
    total_formatted = format_hhmmss(total_time)

    with open(OUTPUT_DIR / "summary.txt", "w", encoding="utf-8") as f_summary:
        f_summary.write("\n".join(summary_lines))

    print(f"✅ Распознавание завершено. Сохранено файлов: {len(wav_files)}")
    print(f"📄 Сводный файл: output/summary.txt")
    print(f"⏱️ Время выполнения: {total_formatted} ({total_time:.2f} секунд)")
    print("   Настройки распознавания:")
    print("\t- Язык: русский (ru)")
    print("\t- Beam size: 5")
    print("\t- VAD-фильтр: включен (threshold=0.5)")
    print("\t- Таймкоды: сохранены")


















# === не реализовано ===
print(" Распознавание речи пока не реализовано до конца.")
print("   🚧 Нет сегментации по голосам")
print("   🚧 Нет деления на реплики и диалоги")
print("   🚧 Нет экспорта в .json или .srt")

# === 4. Завершение ===
print("✅ Выполнение завершено.")
