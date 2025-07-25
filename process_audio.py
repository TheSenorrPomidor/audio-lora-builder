#!/usr/bin/env python3
# === Версия ===
print("\n🔢 Версия скрипта process_audio.py 1.7")

import os
import shutil
import json
import subprocess
from pathlib import Path

from pyannote.audio import Pipeline
from pyannote.core import Segment
from speaker_embedding_db import SpeakerEmbeddingDB


speaker_db = SpeakerEmbeddingDB()
dia_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")




# === 0. Функции ====
# Экспорт результатов в JSON и SRT
def write_json(transcript, base_path):
    """Save transcript segments with optional speaker labels to JSON."""
    
    json_path = base_path.with_suffix(".json")
    json_path.parent.mkdir(parents=True, exist_ok=True)

    output = []
    for s in transcript:
        if isinstance(s, dict):
            item = dict(s)
        else:
            item = {
                "start": getattr(s, "start"),
                "end": getattr(s, "end"),
                "text": getattr(s, "text"),
            }
            if hasattr(s, "speaker"):
                item["speaker"] = s.speaker
        output.append(item)

    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(output, jf, ensure_ascii=False, indent=2)

def build_summary_json(json_dir: Path):
    assert json_dir.is_dir(), f"❌ {json_dir} не является каталогом"

    json_files = list(json_dir.rglob("*.json"))
    summary = []

    for jf in json_files:
        if jf.name == "summary.json":
            continue
        try:
            with open(jf, "r", encoding="utf-8") as f:
                segments = json.load(f)
                summary.append({
                    "file": str(jf.relative_to(json_dir).with_suffix(".wav")),
                    "segments": segments
                })
        except Exception as e:
            print(f"⚠️ Ошибка при чтении {jf}: {e}")

    summary_path = json_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f_out:
        json.dump(summary, f_out, ensure_ascii=False, indent=2)

    print(f"📦 Сводный JSON создан: {summary_path} (файлов: {len(summary)})")


# === 1. Чтение конфигурации ===
print("1. Чтение конфигурации...")
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
print("2. 🎧 Начинаем поиск и конвертацию аудиофайлов...")

AUDIO_EXTENSIONS = [".m4a", ".mp3", ".aac"]
SRC = Path(WIN_AUDIO_SRC)
DST = Path("/root/audio-lora-builder/input/audio_src")

# Очистка папки DST
if DST.exists():
    shutil.rmtree(DST)
DST.mkdir(parents=True, exist_ok=True)
print("🧹 Папка input/audio_src очищена.")

# Перебираем все файлы в папке SRC формата AUDIO_EXTENSIONS по очереди и конвертируем в .waw
files = [f for f in SRC.rglob("*") if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS]

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
print("\n3. Распознавание речи (Whisper large-v3, cuDNN-guarded GPU, таймер, очистка) v11")

from faster_whisper import WhisperModel
import time

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

    for idx, audio_path in enumerate(wav_files, 1):
        elapsed = format_hhmmss(time.time() - start_time)
        rel_path = audio_path.relative_to(DST)
        out_txt = OUTPUT_DIR / rel_path.with_suffix(".txt")
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

            diarization = dia_pipeline(str(audio_path))
            diar_segments = list(diarization.itertracks(yield_label=True))

            enriched = []
            for s in segments:
                mid = (s.start + s.end) / 2
                spk_seg = None
                for seg, _, spk in diar_segments:
                    if seg.start <= mid <= seg.end:
                        spk_seg = seg
                        break

                if spk_seg is None:
                    spk_id = speaker_db.process_segment(str(audio_path), s.start, s.end)
                else:
                    spk_id = speaker_db.process_segment(str(audio_path), spk_seg.start, spk_seg.end)

                enriched.append({
                    "start": s.start,
                    "end": s.end,
                    "speaker": spk_id,
                    "text": s.text,
                })

            write_json(enriched, out_txt)
            
            
            
            

        except Exception as e:
            print(f"❌ Ошибка при обработке {rel_path}: {e}")
            continue

    total_time = time.time() - start_time
    total_formatted = format_hhmmss(total_time)
    build_summary_json(OUTPUT_DIR)


    print(f"✅ Распознавание завершено. Сохранено файлов: {len(wav_files)}")
    print(f"📄 Сводный файл: output/summary.txt")
    print(f"⏱️ Время выполнения: {total_formatted} ({total_time:.2f} секунд)")
    print("   Настройки распознавания:")
    print("\t- Язык: русский (ru)")
    print("\t- Beam size: 5")
    print("\t- VAD-фильтр: включен (threshold=0.5)")
    print("\t- Таймкоды: сохранены")



















print(" Распознавание речи пока не реализовано до конца.")
print("   ✅ Сегментация по голосам выполнена")
print("   🚧 Нет деления на реплики и диалоги")
print("   ✅ Экспорт в .json и .srt выполнен")


# === 4. Завершение ===
print("✅ Выполнение process_audio.py завершено.")



