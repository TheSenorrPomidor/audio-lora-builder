#!/usr/bin/env python3
# === Версия ===
print("\n🔢 Версия скрипта process_audio.py 2.13")

import os
import shutil
import json
import subprocess
import re
import numpy as np
import torch
from pathlib import Path
import time
from collections import defaultdict, Counter
import wave
import contextlib
from sklearn.cluster import KMeans
from tqdm import tqdm

from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from pyannote.audio.core.io import Audio
from pyannote.core import Segment
from pyannote.audio import Model

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

def create_voice_profile(embedding_model, audio_files, min_common_files=3):
    """Create voice profile for the most common speaker"""
    print("\n🔊 Создание профиля голоса...")
    
    # Собираем все эмбеддинги из всех файлов
    all_embeddings = []
    file_embeddings = defaultdict(list)
    audio_reader = Audio(sample_rate=16000, mono=True)
    
    for audio_path in tqdm(audio_files, desc="Извлечение эмбеддингов"):
        try:
            waveform, sample_rate = audio_reader(str(audio_path))
            diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate}, num_speakers=2)
            
            for segment in diarization.itertracks(yield_label=True):
                if isinstance(segment, tuple) and len(segment) == 3:
                    turn, _, speaker = segment
                    segment_audio = waveform[:, int(turn.start * sample_rate):int(turn.end * sample_rate)]
                    
                    # Пропускаем слишком короткие сегменты
                    if segment_audio.shape[1] < 16000 * 0.5:  # Минимум 0.5 секунд
                        continue
                    
                    # Конвертируем в тензор
                    segment_tensor = torch.as_tensor(segment_audio).unsqueeze(0).float()
                    with torch.no_grad():
                        embedding = embedding_model(segment_tensor).numpy()[0]
                    
                    all_embeddings.append(embedding)
                    file_embeddings[str(audio_path)].append(embedding)
        except Exception as e:
            print(f"  ⚠️ Ошибка при обработке {audio_path.name}: {e}")
    
    if not all_embeddings:
        print("⚠️ Не удалось извлечь эмбеддинги")
        return None
    
    # Кластеризация всех эмбеддингов
    X = np.vstack(all_embeddings)
    X = l2_normalize(X)
    
    # Определяем оптимальное количество кластеров
    n_clusters = min(10, max(2, len(all_embeddings) // 20))  # Уменьшено количество кластеров
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X)
    
    # Определяем, какой кластер встречается в наибольшем количестве файлов
    cluster_files = defaultdict(set)
    file_idx = 0
    for file_path, embeddings in file_embeddings.items():
        for _ in embeddings:
            cluster_id = cluster_labels[file_idx]
            cluster_files[cluster_id].add(file_path)
            file_idx += 1
    
    # Выбираем кластер, встречающийся в наибольшем количестве файлов
    best_cluster = None
    max_files = 0
    for cluster_id, files in cluster_files.items():
        if len(files) > max_files:
            max_files = len(files)
            best_cluster = cluster_id
    
    print(f"✅ Профиль создан: кластер {best_cluster} найден в {max_files} файлах")
    
    # Возвращаем центр кластера как эталонный вектор голоса
    return kmeans.cluster_centers_[best_cluster]

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

# === 3. Инициализация моделей ===
print("\n3. ⚙️ Инициализация моделей...")
OUTPUT_DIR = Path("/root/audio-lora-builder/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

wav_files = list(DST.rglob("*.wav"))
if not wav_files:
    print("⚠️ Нет файлов для обработки")
    exit(0)

# Инициализация моделей
whisper_model = WhisperModel(
    "large-v3",
    device="cuda" if torch.cuda.is_available() else "cpu",
    compute_type="float16" if torch.cuda.is_available() else "int8"
)

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=HF_TOKEN
)

embedding_model = Model.from_pretrained("pyannote/embedding")
audio_reader = Audio(sample_rate=16000, mono=True)

# === 4. Создание голосового профиля ===
voice_profile = create_voice_profile(embedding_model, wav_files)

# Если не удалось создать профиль, используем эвристику по длительности
if voice_profile is None:
    print("⚠️ Не удалось создать голосовой профиль, используется эвристика по длительности")
    voice_profile = "duration_fallback"

# === 5. Обработка файлов с использованием профиля ===
print("\n5. 🤖 Распознавание и диаризация с использованием профиля...")
start_all = time.time()
processed_files = 0

for idx, audio_path in enumerate(wav_files, 1):
    rel_path = audio_path.relative_to(DST)
    output_path = OUTPUT_DIR / rel_path.with_suffix(".json")
    
    if output_path.exists():
        print(f"⏩ ({idx}/{len(wav_files)}) Пропуск (уже обработан): {rel_path}")
        continue
        
    print(f"\n📝 ({idx}/{len(wav_files)}) {rel_path}")
    
    try:
        # Загрузка аудио
        waveform, sample_rate = audio_reader(str(audio_path))
        
        # Диаризация
        print("  🎤 Диаризация...")
        diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate}, num_speakers=2)
        
        # Собираем сегменты и их эмбеддинги
        segments = []
        for segment in diarization.itertracks(yield_label=True):
            if isinstance(segment, tuple) and len(segment) == 3:
                turn, _, speaker = segment
                
                # Пропускаем слишком короткие сегменты
                if turn.end - turn.start < 0.5:
                    continue
                    
                segment_audio = waveform[:, int(turn.start * sample_rate):int(turn.end * sample_rate)]
                segment_tensor = torch.as_tensor(segment_audio).unsqueeze(0).float()
                
                with torch.no_grad():
                    embedding = embedding_model(segment_tensor).numpy()[0]
                
                segments.append({
                    "start": turn.start,
                    "end": turn.end,
                    "speaker": speaker,
                    "embedding": embedding
                })
        
        # Определение спикеров с использованием голосового профиля
        if isinstance(voice_profile, np.ndarray):
            for seg in segments:
                # Улучшенное сравнение эмбеддингов
                current_emb = seg["embedding"].reshape(1, -1)
                current_emb_norm = l2_normalize(current_emb)
                profile_norm = l2_normalize(voice_profile.reshape(1, -1))
                
                # Вычисляем косинусное сходство
                similarity = np.dot(current_emb_norm, profile_norm.T)[0][0]
                
                # Динамический порог: чем длиннее сегмент, тем строже проверка
                duration = seg["end"] - seg["start"]
                threshold = 0.5 + min(0.3, duration / 10)  # Динамический порог от 0.5 до 0.8
                
                seg["is_you"] = similarity > threshold
        else:
            # Эвристика по длительности (если не удалось создать профиль)
            total_durations = defaultdict(float)
            for seg in segments:
                duration = seg["end"] - seg["start"]
                total_durations[seg["speaker"]] += duration
            
            if total_durations:
                main_speaker = max(total_durations, key=total_durations.get)
                for seg in segments:
                    seg["is_you"] = seg["speaker"] == main_speaker
        
        # Транскрибация
        print("  📝 Транскрибация...")
        transcriptions, _ = whisper_model.transcribe(
            str(audio_path),
            language="ru",
            beam_size=5,
            vad_filter=True,
            word_timestamps=False
        )
        transcriptions = list(transcriptions)
        
        # Сопоставляем транскрипцию с сегментами
        # Исправление: используем точное сопоставление вместо перекрытия
        for seg in segments:
            seg_text = []
            for t in transcriptions:
                # Проверяем, что сегмент транскрипции полностью внутри сегмента диаризации
                if t.start >= seg["start"] and t.end <= seg["end"]:
                    seg_text.append(t.text)
            
            seg["text"] = " ".join(seg_text).strip()
        
        # Сохранение результатов
        caller_id = extract_phone_number(str(rel_path)) or "caller"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Преобразуем в формат для записи
        json_segments = []
        for seg in segments:
            # Пропускаем пустые сегменты
            if seg["text"] or (seg["end"] - seg["start"]) > 1.0:
                json_segments.append({
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"],
                    "is_you": seg["is_you"]
                })
        
        write_json(json_segments, output_path, rel_path, "0000000000000", caller_id)
        print(f"  💾 Сохранено сегментов: {len(json_segments)} → {output_path}")
        processed_files += 1
        
    except Exception as e:
        print(f"  ❌ Ошибка при обработке файла: {e}")
        continue

total_time = format_hhmmss(time.time() - start_all)
print(f"\n✅ Обработка завершена. Обработано файлов: {processed_files}/{len(wav_files)}")
print(f"⏱️ Время выполнения: {total_time}")

# === Завершение ===
print("\n✅ Выполнение process_audio.py завершено.")