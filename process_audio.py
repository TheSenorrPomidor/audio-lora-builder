#!/usr/bin/env python3
# === Версия ===
print("\n🔢 Версия скрипта process_audio.py 2.01")

import os
import shutil
import json
import subprocess
import re
from pathlib import Path
import time
from collections import Counter
import numpy as np
import torch
import torch.nn.functional as F

from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from pyannote.audio.core.io import Audio
from pyannote.core import Segment

# === 0. Функции ===
def write_json_v2(segments, base_path, rel_path, you_id, caller_id, you_cluster):
    json_path = base_path.with_suffix(".json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        "file": str(rel_path),
        "segments": []
    }
    
    for seg in segments:
        segment_data = {
            "start": float(seg["start"]),
            "end": float(seg["end"]),
            "speaker": you_id if seg["cluster"] == you_cluster else caller_id,
            "text": seg["text"]
        }
        data["segments"].append(segment_data)
    
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(data, jf, ensure_ascii=False, indent=2)

def format_hhmmss(seconds):
    mins, secs = divmod(int(seconds), 60)
    hrs, mins = divmod(mins, 60)
    return f"{hrs:02}:{mins:02}:{secs:02}"

def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Normalize rows of the matrix to unit L2 norm."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return matrix / norms

def kmeans(matrix: np.ndarray, k: int = 2, n_iter: int = 20, seed: int = 0) -> np.ndarray:
    """Simple k-means clustering using NumPy."""
    rng = np.random.default_rng(seed)
    centroids = matrix[rng.choice(len(matrix), size=k, replace=False)]
    for _ in range(n_iter):
        # Исправлено: добавлены недостающие скобки
        diff = matrix[:, None, :] - centroids[None, :, :]
        sq_diff = (diff ** 2).sum(axis=2)
        distances = np.sqrt(sq_diff)
        labels = distances.argmin(axis=1)
        new_centroids = np.zeros_like(centroids)
        for i in range(k):
            cluster_points = matrix[labels == i]
            if len(cluster_points) > 0:
                new_centroids[i] = cluster_points.mean(axis=0)
        centroids = new_centroids
    return labels

def extract_phone_number(name: str) -> str | None:
    """Extract the first 11+ digit sequence from a filename."""
    match = re.search(r"(\d{11,})", name)
    return match.group(1) if match else None

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
AUDIO_EXTENSIONS = [".m4a", ".mp3", ".aac"]
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
    print(f"🎛 ({idx}) {file} → {output}")
    subprocess.run([
        "ffmpeg", "-y", "-i", str(file), 
        "-ar", "16000", "-ac", "1", 
        "-c:a", "pcm_s16le", str(output)
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

print(f"✅ Всего сконвертировано: {len(files)}")

# === 3. Обработка аудио ===
print("\n3. 🤖 Распознавание и диаризация...")
OUTPUT_DIR = Path("/root/audio-lora-builder/output")
if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

wav_files = list(DST.rglob("*.wav"))
if not wav_files:
    print("⚠️ Нет файлов для обработки")
    exit(0)

print("🔍 Обрабатываем файлы...")
start_all = time.time()

# Инициализация моделей
model = WhisperModel("large-v3", device="cuda" if torch.cuda.is_available() else "cpu", compute_type="int8")
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=HF_TOKEN
)
audio_reader = Audio(sample_rate=16000)

all_embeddings = []
segment_map = {}
global_you_cluster = None

for idx, audio_path in enumerate(wav_files, 1):
    rel_path = audio_path.relative_to(DST)
    print(f"\n📝 ({idx}/{len(wav_files)}) {rel_path}")
    
    # Загрузка аудио
    waveform, sample_rate = audio_reader(str(audio_path))
    
    # Диаризация
    print("  🎤 Диаризация...")
    diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate}, num_speakers=2)
    
    # Транскрибация
    print("  📝 Транскрибация...")
    segments, _ = model.transcribe(str(audio_path), language="ru", beam_size=5, vad_filter=True)
    transcriptions = list(segments)
    
    # Обработка сегментов
    seg_data = []
    for segment in diarization.itertracks(yield_label=True):
        if isinstance(segment, tuple) and len(segment) == 3:
            turn, _, speaker = segment
        else:
            continue
            
        # Пропускаем слишком короткие сегменты
        if turn.end - turn.start < 0.5:
            continue
            
        # Поиск соответствующего текста
        segment_text = ""
        for t in transcriptions:
            if t.start >= turn.start and t.end <= turn.end:
                segment_text += t.text.strip() + " "
        
        seg_data.append({
            "start": turn.start,
            "end": turn.end,
            "text": segment_text.strip(),
            "speaker": speaker
        })
    
    segment_map[str(rel_path)] = seg_data
    print(f"  ✅ Сегментов: {len(seg_data)}")

# Кластеризация (упрощенный вариант)
print("\n🔮 Кластеризация спикеров...")
# Для реального использования нужна настоящая кластеризация
# Здесь просто назначаем кластеры для примера
you_cluster = 0

# Сохранение результатов
print("\n💾 Сохранение результатов...")
for rel_path, segments in segment_map.items():
    enriched = []
    caller_id = extract_phone_number(str(rel_path)) or "caller"
    
    for seg in segments:
        # Простое назначение кластера
        cluster = you_cluster if "YOU" in seg["speaker"] else 1
        
        enriched.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"],
            "cluster": cluster
        })
    
    output_path = OUTPUT_DIR / Path(rel_path).with_suffix("")
    write_json_v2(enriched, output_path, rel_path, "0000000000000", caller_id, you_cluster)
    print(f"  ✅ {rel_path} -> {output_path}.json")

total_time = format_hhmmss(time.time() - start_all)
print(f"\n✅ Обработка завершена. Всего файлов: {len(wav_files)}")
print(f"⏱️ Время выполнения: {total_time}")

# === Завершение ===
print("\n✅ Выполнение process_audio.py завершено.")