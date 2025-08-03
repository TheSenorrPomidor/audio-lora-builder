#!/usr/bin/env python3
# === Версия ===
print("\n🔢 Версия скрипта process_audio.py 3.3")

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
import random

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

def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors"""
    a = a.flatten()
    b = b.flatten()
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def create_reliable_voice_profile(embedding_model, audio_files, num_samples=5):
    """Create reliable voice profile using diverse samples"""
    print("\n🔊 Создание надежного профиля голоса...")
    
    # Собираем эмбеддинги из случайных сегментов
    candidate_embeddings = []
    audio_reader = Audio(sample_rate=16000, mono=True)
    device = next(embedding_model.parameters()).device
    
    # Выбираем случайные файлы для профиля
    selected_files = random.sample(audio_files, min(len(audio_files), 3))
    
    for audio_path in selected_files:
        try:
            waveform, sample_rate = audio_reader(str(audio_path))
            diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate}, num_speakers=2)
            
            # Выбираем только сегменты длительностью > 3 секунд
            long_segments = [
                seg for seg in diarization.itertracks(yield_label=True)
                if isinstance(seg, tuple) and len(seg) == 3 and (seg[0].end - seg[0].start) > 3.0
            ]
            
            if not long_segments:
                continue
                
            # Выбираем самый длинный сегмент из файла
            longest_segment = max(long_segments, key=lambda x: x[0].end - x[0].start)
            turn, _, speaker = longest_segment
            
            # Извлекаем эмбеддинг
            segment_audio = waveform[:, int(turn.start * sample_rate):int(turn.end * sample_rate)]
            segment_tensor = torch.as_tensor(segment_audio).unsqueeze(0).float().to(device)
            
            with torch.no_grad():
                embedding = embedding_model(segment_tensor).cpu().numpy()[0]
            
            candidate_embeddings.append(embedding)
        except Exception as e:
            print(f"  ⚠️ Ошибка при обработке {audio_path.name}: {e}")
    
    if not candidate_embeddings:
        print("⚠️ Не удалось собрать эталонные эмбеддинги")
        return None
    
    # Усредняем эмбеддинги для создания профиля
    profile = np.mean(candidate_embeddings, axis=0)
    print(f"✅ Профиль создан на основе {len(candidate_embeddings)} эталонных сегментов")
    
    return profile

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

# Инициализация диаризации
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=HF_TOKEN
)

# Инициализируем модель эмбеддингов
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embedding_model = Model.from_pretrained("pyannote/embedding").to(device)
audio_reader = Audio(sample_rate=16000, mono=True)

# === 4. Создание голосового профиля ===
voice_profile = create_reliable_voice_profile(embedding_model, wav_files)

if voice_profile is None:
    print("⚠️ Не удалось создать голосовой профиль, будет использоваться только диаризация")
    voice_profile = None

# Инициализация Whisper на CPU
print("  ⚙️ Инициализация Whisper на CPU...")
whisper_model = WhisperModel(
    "large-v3",
    device="cpu",
    compute_type="int8"
)

# === 5. Обработка файлов ===
print("\n5. 🤖 Распознавание и диаризация...")
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
        
        # Собираем сегменты
        segments = []
        for segment in diarization.itertracks(yield_label=True):
            if isinstance(segment, tuple) and len(segment) == 3:
                turn, _, speaker = segment
                
                # Пропускаем слишком короткие сегменты
                if turn.end - turn.start < 0.5:
                    continue
                    
                segments.append({
                    "start": turn.start,
                    "end": turn.end,
                    "speaker": speaker,
                })
        
        # Если есть голосовой профиль, вычисляем эмбеддинги и определяем "я"
        if voice_profile is not None:
            print("  🔍 Определение спикеров с помощью профиля...")
            for seg in segments:
                segment_audio = waveform[:, int(seg["start"] * sample_rate):int(seg["end"] * sample_rate)]
                segment_tensor = torch.as_tensor(segment_audio).unsqueeze(0).float().to(device)
                
                with torch.no_grad():
                    embedding = embedding_model(segment_tensor).cpu().numpy()[0]
                
                # Вычисляем косинусное сходство
                similarity = cosine_similarity(embedding, voice_profile)
                seg["is_you"] = similarity > 0.5
                
                print(f"    Сегмент {seg['start']:.2f}-{seg['end']:.2f}: "
                      f"сходство={similarity:.2f}, is_you={seg['is_you']}")
        else:
            print("  ⚠️ Голосовой профиль недоступен, используется простая эвристика")
            # Эвристика: первый спикер - собеседник, второй - вы
            speakers = {seg["speaker"] for seg in segments}
            if len(speakers) == 2:
                speaker_roles = {list(speakers)[0]: False, list(speakers)[1]: True}
                for seg in segments:
                    seg["is_you"] = speaker_roles[seg["speaker"]]
            else:
                # Если не удалось определить 2 спикеров, помечаем все как собеседника
                for seg in segments:
                    seg["is_you"] = False
        
        # Транскрибация на CPU
        print("  📝 Транскрибация на CPU...")
        try:
            transcriptions, _ = whisper_model.transcribe(
                str(audio_path),
                language="ru",
                beam_size=5,
                vad_filter=True,
                word_timestamps=False
            )
            transcriptions = list(transcriptions)
            print(f"  🔠 Распознано сегментов: {len(transcriptions)}")
        except Exception as e:
            print(f"  ❌ Ошибка транскрибации: {e}")
            transcriptions = []
        
        # Сопоставляем транскрипцию с сегментами
        for seg in segments:
            seg_text = []
            best_match = ""
            best_overlap = 0
            
            for t in transcriptions:
                # Рассчитываем перекрытие
                overlap_start = max(t.start, seg["start"])
                overlap_end = min(t.end, seg["end"])
                overlap_duration = max(0, overlap_end - overlap_start)
                
                # Рассчитываем процент перекрытия
                seg_duration = seg["end"] - seg["start"]
                if seg_duration > 0:
                    overlap_percent = overlap_duration / seg_duration
                else:
                    overlap_percent = 0
                
                # Выбираем лучший вариант
                if overlap_percent > best_overlap:
                    best_overlap = overlap_percent
                    best_match = t.text
            
            # Если найдено хорошее соответствие, используем текст
            if best_overlap > 0.3:
                seg["text"] = best_match
            else:
                seg["text"] = ""
        
        # Сохранение результатов
        caller_id = extract_phone_number(str(rel_path)) or "caller"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        write_json(segments, output_path, rel_path, "0000000000000", caller_id)
        print(f"  💾 Сохранено {len(segments)} сегментов → {output_path}")
        processed_files += 1
        
    except Exception as e:
        print(f"  ❌ Критическая ошибка при обработке файла: {e}")
        continue

total_time = format_hhmmss(time.time() - start_all)
print(f"\n✅ Обработка завершена. Обработано файлов: {processed_files}/{len(wav_files)}")
print(f"⏱️ Время выполнения: {total_time}")

# === Завершение ===
print("\n✅ Выполнение process_audio.py завершено.")