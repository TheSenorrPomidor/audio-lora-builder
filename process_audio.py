#!/usr/bin/env python3
# === Версия ===
print("\n🔢 Версия скрипта process_audio.py 2.14 (Stable GPU)")

import os
import shutil
import json
import subprocess
import re
import numpy as np
import torch
from pathlib import Path
import time
from collections import defaultdict
import wave
import contextlib

from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from pyannote.audio.core.io import Audio
from pyannote.core import Segment

# === 0. Функции ===
def write_json(segments, json_path, rel_path, you_id, caller_id):
    json_path.parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        "file": str(rel_path),
        "segments": []
    }
    
    for seg in segments:
        segment_data = {
            "start": float(seg["start"]),
            "end": float(seg["end"]),
            "speaker": you_id if seg["is_you"] else caller_id,
            "text": seg["text"]
        }
        data["segments"].append(segment_data)
    
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(data, jf, ensure_ascii=False, indent=2)

def format_hhmmss(seconds):
    mins, secs = divmod(int(seconds), 60)
    hrs, mins = divmod(mins, 60)
    return f"{hrs:02}:{mins:02}:{secs:02}"

def extract_phone_number(name: str) -> str | None:
    """Extract the first 11+ digit sequence from a filename."""
    match = re.search(r"(\d{11,})", name)
    return match.group(1) if match else None

def get_audio_duration(wav_path):
    """Get duration of audio file in seconds"""
    with contextlib.closing(wave.open(str(wav_path), 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        return frames / float(rate)

def l2_normalize(embeddings):
    """Normalize embeddings to unit length"""
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return embeddings / norms

def assign_speaker_roles(diarization, audio_duration):
    """Identify which speaker is 'you' based on speech duration"""
    speaker_durations = defaultdict(float)
    
    for segment in diarization.itertracks(yield_label=True):
        if isinstance(segment, tuple) and len(segment) == 3:
            turn, _, speaker = segment
            duration = turn.end - turn.start
            speaker_durations[speaker] += duration
    
    if not speaker_durations:
        return {}
    
    # The speaker with the longest total duration is "you"
    you_speaker = max(speaker_durations, key=speaker_durations.get)
    return {speaker: (speaker == you_speaker) for speaker in speaker_durations}

# === 1. Чтение конфигурации ===
print("1. Чтение конфигурации...")
ENV_FILE = "/root/audio-lora-builder/config/env.vars"
WIN_AUDIO_SRC = None
HF_TOKEN = os.getenv("HF_TOKEN")

if os.path.exists(ENV_FILE):
    with open(ENV_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("WIN_AUDIO_SRC="):
                WIN_AUDIO_SRC = line.strip().split("=", 1)[1]
            elif line.startswith("HF_TOKEN=") and not HF_TOKEN:
                HF_TOKEN = line.strip().split("=", 1)[1]

if not WIN_AUDIO_SRC or not os.path.exists(WIN_AUDIO_SRC):
    print("⚠️ Не удалось найти корректный путь до папки с аудиофайлами.")
    user_input = input("Введите путь до папки с аудиофайлами [/mnt/c/]: ").strip()
    WIN_AUDIO_SRC = user_input or "/mnt/c/"
    WIN_AUDIO_SRC = WIN_AUDIO_SRC.replace("\\", "/")
    
    Path("/root/audio-lora-builder/config").mkdir(parents=True, exist_ok=True)
    with open(ENV_FILE, "w", encoding="utf-8") as f:
        f.write(f"WIN_AUDIO_SRC={WIN_AUDIO_SRC}\n")
        if HF_TOKEN:
            f.write(f"HF_TOKEN={HF_TOKEN}\n")

if not HF_TOKEN:
    print("⚠️ Не найден токен Hugging Face.")
    HF_TOKEN = input("Введите токен Hugging Face: ").strip()
    Path("/root/audio-lora-builder/config").mkdir(parents=True, exist_ok=True)
    with open(ENV_FILE, "a", encoding="utf-8") as f:
        f.write(f"HF_TOKEN={HF_TOKEN}\n")

# === 2. Конвертация файлов ===
print("2. 🎧 Конвертация аудиофайлов...")
AUDIO_EXTENSIONS = [".m4a", ".mp3", ".aac", ".wav"]
SRC = Path(WIN_AUDIO_SRC)
DST = Path("/root/audio-lora-builder/input/audio_src")

if DST.exists():
    shutil.rmtree(DST)
DST.mkdir(parents=True, exist_ok=True)

files = [f for f in SRC.rglob("*") if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS]
for idx, file in enumerate(files, 1):
    relative = file.relative_to(SRC)
    output = DST / relative.with_suffix(".wav")
    output.parent.mkdir(parents=True, exist_ok=True)
    
    if not output.exists() or file.stat().st_mtime > output.stat().st_mtime:
        print(f"🎛 ({idx}) {file} → {output}")
        subprocess.run([
            "ffmpeg", "-y", "-i", str(file), 
            "-ar", "16000", "-ac", "1", 
            "-c:a", "pcm_s16le", str(output)
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        print(f"⏩ ({idx}) Пропуск (уже сконвертирован): {file}")

print(f"✅ Всего аудиофайлов: {len(files)}")

# === 3. Обработка аудио ===
print("\n3. 🤖 Распознавание и диаризация...")
OUTPUT_DIR = Path("/root/audio-lora-builder/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

wav_files = list(DST.rglob("*.wav"))
if not wav_files:
    print("⚠️ Нет файлов для обработки")
    exit(0)

print("🔍 Обрабатываем файлы...")
start_all = time.time()

# Инициализация моделей
whisper_model = WhisperModel(
    "large-v3",
    device="cuda" if torch.cuda.is_available() else "cpu",
    compute_type="float16"  # Используем float16 для лучшей совместимости
)

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=HF_TOKEN
)
audio_reader = Audio(sample_rate=16000, mono=True)

processed_files = 0

for idx, audio_path in enumerate(wav_files, 1):
    rel_path = audio_path.relative_to(DST)
    output_path = OUTPUT_DIR / rel_path.with_suffix(".json")
    
    if output_path.exists():
        print(f"⏩ ({idx}/{len(wav_files)}) Пропуск (уже обработан): {rel_path}")
        continue
        
    print(f"\n📝 ({idx}/{len(wav_files)}) {rel_path}")
    
    try:
        # Получаем длительность аудио
        audio_duration = get_audio_duration(audio_path)
        
        # Загрузка аудио
        waveform, sample_rate = audio_reader(str(audio_path))
        
        # Диаризация
        print("  🎤 Диаризация...")
        diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate}, num_speakers=2)
        
        # Определяем кто есть кто
        speaker_roles = assign_speaker_roles(diarization, audio_duration)
        if not speaker_roles:
            print("  ⚠️ Не удалось определить спикеров")
            continue
            
        # Транскрибация
        print("  📝 Транскрибация...")
        segments, info = whisper_model.transcribe(
            str(audio_path),
            language="ru",
            beam_size=5,
            vad_filter=True,
            word_timestamps=False
        )
        transcriptions = list(segments)
        print(f"  🔠 Распознано сегментов: {len(transcriptions)}")
        
        # Сопоставляем диаризацию с транскрипцией
        enriched_segments = []
        
        # Собираем все сегменты диаризации
        diarization_segments = []
        for segment in diarization.itertracks(yield_label=True):
            if isinstance(segment, tuple) and len(segment) == 3:
                turn, _, speaker = segment
                diarization_segments.append({
                    "start": turn.start,
                    "end": turn.end,
                    "speaker": speaker,
                    "is_you": speaker_roles.get(speaker, False),
                    "text_candidates": []
                })
        
        # Сортируем по времени начала
        diarization_segments.sort(key=lambda x: x["start"])
        
        # Сопоставляем текстовые сегменты с диаризацией
        for t in transcriptions:
            # Ищем лучший сегмент диаризации по перекрытию
            best_overlap = 0
            best_segment = None
            
            for d_seg in diarization_segments:
                # Рассчитываем перекрытие
                overlap_start = max(t.start, d_seg["start"])
                overlap_end = min(t.end, d_seg["end"])
                overlap_duration = max(0, overlap_end - overlap_start)
                
                # Рассчитываем процент перекрытия
                t_duration = t.end - t.start
                overlap_percent = overlap_duration / t_duration if t_duration > 0 else 0
                
                if overlap_percent > best_overlap:
                    best_overlap = overlap_percent
                    best_segment = d_seg
            
            # Добавляем текст к лучшему сегменту
            if best_segment and best_overlap > 0.3:  # Минимум 30% перекрытия
                best_segment["text_candidates"].append(t.text)
        
        # Формируем окончательные сегменты
        for d_seg in diarization_segments:
            if d_seg["text_candidates"]:
                # Объединяем тексты
                combined_text = " ".join(d_seg["text_candidates"])
                
                enriched_segments.append({
                    "start": d_seg["start"],
                    "end": d_seg["end"],
                    "text": combined_text.strip(),
                    "is_you": d_seg["is_you"]
                })
        
        # Сохранение результатов
        caller_id = extract_phone_number(str(rel_path)) or "caller"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(enriched_segments, output_path, rel_path, "0000000000000", caller_id)
        print(f"  💾 Сохранено сегментов: {len(enriched_segments)} → {output_path}")
        processed_files += 1
        
    except Exception as e:
        print(f"  ❌ Ошибка при обработке файла: {e}")
        continue

total_time = format_hhmmss(time.time() - start_all)
print(f"\n✅ Обработка завершена. Обработано файлов: {processed_files}/{len(wav_files)}")
print(f"⏱️ Время выполнения: {total_time}")

# === Завершение ===
print("\n✅ Выполнение process_audio.py завершено.")