#!/usr/bin/env python3
# === Версия ===
print("\n🔢 Версия скрипта process_audio.py 2.45 (Stable GPU)")

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
import traceback
from sklearn.cluster import KMeans

from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from pyannote.audio.core.io import Audio
from pyannote.core import Segment
from pyannote.audio import Inference
from pyannote.core import SlidingWindowFeature  # Добавлен импорт для обработки типа

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

def average_pairwise_similarity(embeddings):
    """Calculate average cosine similarity for embeddings"""
    n = len(embeddings)
    if n < 2:
        return 0.0
    
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i+1, n):
            total += np.dot(embeddings[i], embeddings[j])
            count += 1
            
    return total / count if count > 0 else 0.0

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
    user_input = input("Введите пути до папки с аудиофайлами [/mnt/c/]: ").strip()
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
    compute_type="float16"
)

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=HF_TOKEN
)
audio_reader = Audio(sample_rate=16000, mono='downmix')

# Модель для извлечения эмбеддингов
embedding_model = Inference(
    "pyannote/embedding",
    pre_aggregation_hook=lambda spec: np.mean(spec, axis=-1)
)

# === Первый проход: извлечение эмбеддингов ===
print("🔍 Первый проход: извлечение голосовых эмбеддингов...")
all_embeddings = []
all_file_names = []
all_speaker_keys = []
diarization_data = {}
embedding_dim = None  # Для проверки размерности

for idx, audio_path in enumerate(wav_files, 1):
    rel_path = audio_path.relative_to(DST)
    output_path = OUTPUT_DIR / rel_path.with_suffix(".json")
    
    if output_path.exists():
        print(f"⏩ ({idx}/{len(wav_files)}) Пропуск (уже обработан): {rel_path}")
        continue
        
    print(f"  🎤 ({idx}/{len(wav_files)}) {rel_path} (извлечение эмбеддингов)")
    
    try:
        # Получаем длительность файла для проверки границ
        file_duration = get_audio_duration(audio_path)
        
        # Диаризация
        diarization = pipeline(str(audio_path), num_speakers=2)
        
        # Сохраняем данные диаризации
        file_segments = []
        speaker_embeddings = defaultdict(list)
        
        for segment in diarization.itertracks(yield_label=True):
            if isinstance(segment, tuple) and len(segment) == 3:
                turn, _, speaker = segment
                seg = Segment(turn.start, turn.end)
                
                # Корректируем границы сегмента
                seg = Segment(
                    max(0, min(seg.start, file_duration - 0.01)),
                    min(file_duration, max(seg.end, 0.01))
                )
                
                # Пропускаем слишком короткие или некорректные сегменты
                if seg.duration < 0.1 or seg.end <= seg.start:  # 100 ms
                    continue
                
                file_segments.append((seg.start, seg.end, speaker))
                
                # Извлекаем эмбеддинг для сегмента
                try:
                    # ПРАВИЛЬНЫЙ ВЫЗОВ: crop(file, segment)
                    waveform, sample_rate = audio_reader.crop(
                        str(audio_path),  # Путь к файлу
                        seg                # Объект Segment
                    )
                    
                    # Явное преобразование в numpy-массив
                    if isinstance(waveform, torch.Tensor):
                        waveform = waveform.numpy()
                    
                    # Проверка на пустой сегмент
                    if waveform.size == 0:
                        print(f"    ⚠️ Пустой сегмент ({seg.duration:.2f}s), пропускаем")
                        continue
                    
                    # Обработка нормализации
                    try:
                        # Нормализуем аудио
                        max_val = np.max(np.abs(waveform))
                        if max_val > 0:
                            waveform = waveform / max_val
                    except Exception as e:
                        tb = traceback.extract_tb(e.__traceback__)[0]
                        print(f"    ⚠️ Ошибка нормализации: {e}, файл {__file__}, строка {tb.lineno}")
                        continue
                    
                    # Преобразуем в torch.Tensor
                    tensor = torch.from_numpy(waveform).float()
                    
                    # Обеспечиваем правильную размерность (каналы, время)
                    if tensor.ndim == 1:
                        tensor = tensor.unsqueeze(0)  # (1, time)
                    
                    # Извлекаем эмбеддинг
                    embedding_result = embedding_model({
                        "waveform": tensor, 
                        "sample_rate": sample_rate
                    })
                    
                    # Обработка разных типов возвращаемых значений
                    if isinstance(embedding_result, SlidingWindowFeature):
                        embedding = embedding_result.data.squeeze()
                    elif isinstance(embedding_result, torch.Tensor):
                        embedding = embedding_result.cpu().numpy().squeeze()
                    elif isinstance(embedding_result, np.ndarray):
                        embedding = embedding_result.squeeze()
                    else:
                        print(f"    ⚠️ Неизвестный тип эмбеддинга: {type(embedding_result)}")
                        continue
                    
                    # Проверяем размерность эмбеддингов
                    if embedding_dim is None:
                        embedding_dim = embedding.shape[0]
                    
                    if embedding.shape[0] != embedding_dim:
                        print(f"    ⚠️ Неправильная размерность эмбеддинга: {embedding.shape} (ожидалось {embedding_dim}), пропускаем")
                        continue
                    
                    speaker_embeddings[speaker].append(embedding)
                except Exception as e:
                    # Улучшенная обработка ошибок с указанием номера строки
                    tb = traceback.extract_tb(e.__traceback__)[0]
                    print(f"    ⚠️ Ошибка при обработке сегмента: {e}, файл {__file__}, строка {tb.lineno}")
                    continue
        
        diarization_data[audio_path] = file_segments
        
        # Усредняем эмбеддинги по спикерам
        for speaker, embeddings_list in speaker_embeddings.items():
            if embeddings_list:
                # Проверяем одинаковую размерность всех эмбеддингов
                dims = [e.shape[0] for e in embeddings_list]
                if len(set(dims)) > 1:
                    print(f"    ⚠️ Разные размерности эмбеддингов для спикера {speaker}: {dims}")
                    continue
                
                avg_embedding = np.mean(embeddings_list, axis=0)
                avg_embedding = l2_normalize(avg_embedding).flatten()
                all_embeddings.append(avg_embedding)
                all_file_names.append(audio_path.name)
                all_speaker_keys.append(speaker)
            
    except Exception as e:
        # Улучшенная обработка ошибок с указанием номера строки
        tb = traceback.extract_tb(e.__traceback__)[0]
        print(f"  ❌ Ошибка при извлечении эмбеддингов: {e}, файл {__file__}, строка {tb.lineno}")

# Проверка наличия эмбеддингов
if not all_embeddings:
    print("⚠️ Не удалось извлечь эмбеддинги, выход")
    exit(1)

print(f"🔮 Извлечено эмбеддингов: {len(all_embeddings)}")

# Проверяем размерность всех эмбеддингов
emb_dims = [e.shape[0] for e in all_embeddings] if all_embeddings[0].ndim == 1 else [e.size for e in all_embeddings]
if len(set(emb_dims)) > 1:
    print(f"⚠️ Обнаружены эмбеддинги разной размерности: {set(emb_dims)}")
    # Оставляем только эмбеддинги с наиболее частой размерностью
    dim_counts = {dim: emb_dims.count(dim) for dim in set(emb_dims)}
    common_dim = max(dim_counts, key=dim_counts.get)
    filtered_embeddings = [e for e in all_embeddings if (e.shape[0] if e.ndim == 1 else e.size) == common_dim]
    print(f"🔮 Оставляем {len(filtered_embeddings)}/{len(all_embeddings)} эмбеддингов размерностью {common_dim}")
    all_embeddings = filtered_embeddings

if len(all_embeddings) < 2:
    print("⚠️ Недостаточно эмбеддингов для кластеризации (требуется минимум 2)")
    exit(1)

print("🔮 Кластеризация спикеров...")
embeddings_array = np.array(all_embeddings)

# Проверка, что эмбеддингов достаточно для кластеризации
if len(embeddings_array) < 2:
    print("⚠️ Недостаточно данных для кластеризации (требуется минимум 2 эмбеддинга)")
    exit(1)

kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(embeddings_array)
labels = kmeans.labels_

# Определяем кластер для "я" по внутрикластерному сходству
cluster0_mask = (labels == 0)
cluster1_mask = (labels == 1)

cluster0_emb = embeddings_array[cluster0_mask]
cluster1_emb = embeddings_array[cluster1_mask]

sim0 = average_pairwise_similarity(cluster0_emb)
sim1 = average_pairwise_similarity(cluster1_emb)

print(f"🔮 Сходство кластера 0: {sim0:.4f}, кластера 1: {sim1:.4f}")

if sim0 > sim1:
    me_cluster = 0
    print("🔮 Кластер 0 идентифицирован как 'я'")
else:
    me_cluster = 1
    print("🔮 Кластер 1 идентифицирован как 'я'")

# Создаем словарь для определения ролей
speaker_roles = {}
for i in range(len(all_embeddings)):
    file_name = all_file_names[i]
    speaker_key = all_speaker_keys[i]
    cluster_id = labels[i]
    speaker_roles[(file_name, speaker_key)] = (cluster_id == me_cluster)

# === Второй проход: транскрипция ===
print("🔍 Второй проход: транскрипция и формирование JSON...")
processed_files = 0

for idx, audio_path in enumerate(wav_files, 1):
    rel_path = audio_path.relative_to(DST)
    output_path = OUTPUT_DIR / rel_path.with_suffix(".json")
    
    if output_path.exists():
        print(f"⏩ ({idx}/{len(wav_files)}) Пропуск (уже обработан): {rel_path}")
        continue
        
    # Пропускаем файлы без данных диаризации
    if audio_path not in diarization_data:
        print(f"  ⚠️ ({idx}/{len(wav_files)}) Пропуск (нет данных диаризации): {rel_path}")
        continue
        
    print(f"\n📝 ({idx}/{len(wav_files)}) {rel_path}")
    
    try:
        file_segments = diarization_data[audio_path]
        
        # Создаем структуру для обогащенных сегментов
        diarization_segments = []
        for start, end, speaker in file_segments:
            is_you = speaker_roles.get((audio_path.name, speaker), False)
            diarization_segments.append({
                "start": start,
                "end": end,
                "speaker": speaker,
                "is_you": is_you,
                "text_candidates": []
            })
        
        # Транскрибация
        segments, info = whisper_model.transcribe(
            str(audio_path),
            language="ru",
            beam_size=5,
            vad_filter=True,
            word_timestamps=False
        )
        transcriptions = list(segments)
        print(f"  🔠 Распознано сегментов: {len(transcriptions)}")
        
        # Сопоставляем транскрипции с диаризацией
        for t in transcriptions:
            best_overlap = 0
            best_segment = None
            
            for d_seg in diarization_segments:
                overlap_start = max(t.start, d_seg["start"])
                overlap_end = min(t.end, d_seg["end"])
                overlap_duration = max(0, overlap_end - overlap_start)
                t_duration = t.end - t.start
                overlap_percent = overlap_duration / t_duration if t_duration > 0 else 0
                
                if overlap_percent > best_overlap:
                    best_overlap = overlap_percent
                    best_segment = d_seg
            
            if best_segment and best_overlap > 0.3:
                best_segment["text_candidates"].append(t.text)
        
        # Формируем окончательные сегменты
        enriched_segments = []
        for d_seg in diarization_segments:
            if d_seg["text_candidates"]:
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
        # Улучшенная обработка ошибок с указанием номера строки
        tb = traceback.extract_tb(e.__traceback__)[0]
        print(f"  ❌ Ошибка при обработке файла: {e}, файл {__file__}, строка {tb.lineno}")

total_time = format_hhmmss(time.time() - start_all)
print(f"\n✅ Обработка завершена. Обработано файлов: {processed_files}/{len(wav_files)}")
print(f"⏱️ Время выполнения: {total_time}")

# === Завершение ===
print("\n✅ Выполнение process_audio.py завершено.")