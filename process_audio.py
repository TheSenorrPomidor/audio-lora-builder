#!/usr/bin/env python3
# === Версия ===
print("\n🔢 Версия скрипта process_audio.py 2.90 (Optimized Speaker Recognition)")

###############################################################################
# НАСТРАИВАЕМЫЕ ПАРАМЕТРЫ
###############################################################################

# === Параметры определения спикеров ===
SIMILARITY_THRESHOLD = 0.75       # Порог сходства голосов (0.7-0.8). Чем выше, тем строже сравнение
MIN_SPEAKER_DURATION = 0.2        # Минимальная длительность сегмента для обработки (секунды)
MAIN_CLUSTER_BIAS = 0.05          # Смещение в пользу основного кластера (0.01-0.1)

# === Параметры VAD (Voice Activity Detection) ===
VAD_ONSET = 0.35                  # Порог начала речи (0.3-0.5)
VAD_OFFSET = 0.35                 # Порог окончания речи (0.3-0.5)
MIN_SPEECH_DURATION_MS = 150      # Минимальная длительность речи (мс)

# === Параметры транскрипции ===
WHISPER_BEAM_SIZE = 5             # Качество распознавания (3-5). Больше = лучше, но медленнее
WHISPER_LANGUAGE = "ru"           # Язык распознавания
SUPPRESS_EMPTY = True             # Подавлять пустые сегменты (True/False)

# === Параметры объединения сегментов ===
MERGE_MIN_DURATION = 0.5          # Минимальная длительность для объединения (секунды)
MERGE_MAX_GAP = 0.5               # Максимальный разрыв для объединения (секунды)

# === Параметры цензуры ===
ALLOW_PROFANITY = True            # Разрешить ненормативную лексику (True/False)
if ALLOW_PROFANITY:
    WHISPER_SUPPRESS_TOKENS = []  # Пустой список = разрешить все слова
else:
    WHISPER_SUPPRESS_TOKENS = None  # Стандартная цензура

###############################################################################
# ОСНОВНОЙ КОД СКРИПТА
###############################################################################

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
from sklearn.decomposition import PCA
from scipy.spatial.distance import cosine
import pickle

from faster_whisper import WhisperModel
from faster_whisper.vad import VadOptions
from pyannote.audio import Pipeline
from pyannote.audio.core.io import Audio
from pyannote.core import Segment
from pyannote.audio import Inference
from pyannote.core import SlidingWindowFeature

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
            similarity = 1 - cosine(embeddings[i], embeddings[j])
            total += similarity
            count += 1
            
    return total / count if count > 0 else 0.0

def merge_short_segments(segments, min_duration=MERGE_MIN_DURATION, max_gap=MERGE_MAX_GAP):
    """Объединяет короткие сегменты с учетом пауз"""
    if not segments:
        return []
    
    merged = []
    current = segments[0].copy()
    
    for seg in segments[1:]:
        gap = seg["start"] - current["end"]
        same_speaker = current["is_you"] == seg["is_you"]
        
        if same_speaker and gap < max_gap and (current["end"] - current["start"] < min_duration or 
                             seg["end"] - seg["start"] < min_duration):
            current["end"] = seg["end"]
            current["text"] = (current.get("text", "") + " " + seg.get("text", "")).strip()
        else:
            merged.append(current)
            current = seg.copy()
    
    merged.append(current)
    return merged

# === Глобальный словарь известных собеседников ===
KNOWN_CALLERS_FILE = Path("/root/audio-lora-builder/config/known_callers.pkl")
known_caller_ids = {}  # caller_id -> embedding

if KNOWN_CALLERS_FILE.exists():
    try:
        with open(KNOWN_CALLERS_FILE, "rb") as f:
            known_caller_ids = pickle.load(f)
        print(f"🔊 Загружено {len(known_caller_ids)} известных собеседников")
    except:
        print("⚠️ Не удалось загрузить известных собеседников")

def save_known_callers():
    with open(KNOWN_CALLERS_FILE, "wb") as f:
        pickle.dump(known_caller_ids, f)

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

# === 2. Конвертация файлов (с нормализацией громкости) ===
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
            "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
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
embedding_dim = None

# Сохраняем усредненные эмбеддинги для каждого спикера
file_speaker_avg_embeddings = {}  # (file_name, speaker) -> avg_embedding
file_to_speakers = defaultdict(set)

for idx, audio_path in enumerate(wav_files, 1):
    rel_path = audio_path.relative_to(DST)
    output_path = OUTPUT_DIR / rel_path.with_suffix(".json")
    
    if output_path.exists():
        print(f"⏩ ({idx}/{len(wav_files)}) Пропуск (уже обработан): {rel_path}")
        continue
        
    print(f"  🎤 ({idx}/{len(wav_files)}) {rel_path} (извлечение эмбеддингов)")
    
    try:
        file_duration = get_audio_duration(audio_path)
        
        try:
            diarization = pipeline(str(audio_path), num_speakers=2)
        except Exception as e:
            print(f"    ⚠️ Ошибка диаризации: {e}. Повторная попытка с num_speakers=2")
            try:
                diarization = pipeline(str(audio_path), num_speakers=2)
            except:
                print(f"    ❌ Повторная ошибка диаризации. Используем fallback подход")
                diarization = []
                segment = Segment(0, file_duration)
                diarization.append((segment, "SPEAKER_00"))
        
        file_segments = []
        speaker_embeddings = defaultdict(list)
        file_speakers = set()
        
        for segment in diarization.itertracks(yield_label=True):
            if isinstance(segment, tuple) and len(segment) == 3:
                turn, _, speaker = segment
                seg = Segment(turn.start, turn.end)
                
                seg = Segment(
                    max(0, min(seg.start, file_duration - 0.01)),
                    min(file_duration, max(seg.end, 0.01))
                )
                
                if seg.duration < MIN_SPEAKER_DURATION or seg.end <= seg.start:
                    continue
                
                file_segments.append((seg.start, seg.end, speaker))
                file_speakers.add(speaker)
                
                try:
                    waveform, sample_rate = audio_reader.crop(
                        str(audio_path),
                        seg
                    )
                    
                    if isinstance(waveform, torch.Tensor):
                        waveform = waveform.numpy()
                    
                    if waveform.size == 0:
                        print(f"    ⚠️ Пустой сегмент ({seg.duration:.2f}s), пропускаем")
                        continue
                    
                    try:
                        max_val = np.max(np.abs(waveform))
                        if max_val > 0:
                            waveform = waveform / max_val
                    except Exception as e:
                        tb = traceback.extract_tb(e.__trace_back__)[0]
                        print(f"    ⚠️ Ошибка нормализации: {e}, файл {__file__}, строка {tb.lineno}")
                        continue
                    
                    tensor = torch.from_numpy(waveform).float()
                    
                    if tensor.ndim == 1:
                        tensor = tensor.unsqueeze(0)
                    
                    embedding_result = embedding_model({
                        "waveform": tensor, 
                        "sample_rate": sample_rate
                    })
                    
                    if isinstance(embedding_result, SlidingWindowFeature):
                        embedding = embedding_result.data.squeeze()
                    elif isinstance(embedding_result, torch.Tensor):
                        embedding = embedding_result.cpu().numpy().squeeze()
                    elif isinstance(embedding_result, np.ndarray):
                        embedding = embedding_result.squeeze()
                    else:
                        print(f"    ⚠️ Неизвестный тип эмбеддинга: {type(embedding_result)}")
                        continue
                    
                    if embedding.ndim > 1:
                        embedding = np.mean(embedding, axis=0)
                    
                    if embedding_dim is None:
                        embedding_dim = embedding.shape[0]
                    
                    if embedding.shape[0] != embedding_dim:
                        print(f"    ⚠️ Неправильная размерность эмбеддинга: {embedding.shape} (ожидалось {embedding_dim}), пропускаем")
                        continue
                    
                    speaker_embeddings[speaker].append(embedding)
                except Exception as e:
                    tb = traceback.extract_tb(e.__traceback__)[0]
                    print(f"    ⚠️ Ошибка при обработке сегмента: {e}, файл {__file__}, строка {tb.lineno}")
                    continue
        
        diarization_data[audio_path] = file_segments
        file_to_speakers[audio_path.name] = file_speakers
        
        # Сохраняем усредненные эмбеддинги для каждого спикера
        for speaker, embeddings_list in speaker_embeddings.items():
            if embeddings_list:
                dims = [e.shape[0] for e in embeddings_list]
                if len(set(dims)) > 1:
                    print(f"    ⚠️ Разные размерности эмбеддингов для спикера {speaker}: {dims}")
                    continue
                
                avg_embedding = np.mean(embeddings_list, axis=0)
                avg_embedding = l2_normalize(avg_embedding).flatten()
                
                # Сохраняем для каждого спикера в файле
                file_speaker_avg_embeddings[(audio_path.name, speaker)] = avg_embedding
                
                all_embeddings.append(avg_embedding)
                all_file_names.append(audio_path.name)
                all_speaker_keys.append(speaker)
            
    except Exception as e:
        tb = traceback.extract_tb(e.__traceback__)[0]
        print(f"  ❌ Ошибка при извлечении эмбеддингов: {e}, файл {__file__}, строка {tb.lineno}")

if not all_embeddings:
    print("⚠️ Не удалось извлечь эмбеддинги, выход")
    exit(1)

print(f"🔮 Извлечено эмбеддингов: {len(all_embeddings)}")

emb_dims = [e.shape[0] for e in all_embeddings] if all_embeddings[0].ndim == 1 else [e.size for e in all_embeddings]
if len(set(emb_dims)) > 1:
    print(f"⚠️ Обнаружены эмбеддинги разной размерности: {set(emb_dims)}")
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

pca = PCA(n_components=min(50, len(embeddings_array)-1))
embeddings_reduced = pca.fit_transform(embeddings_array)

kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(embeddings_reduced)
labels = kmeans.labels_

# Определение основного кластера "я" по охвату файлов
print("🔮 Определение основного кластера 'я'...")
cluster_files = {0: set(), 1: set()}
for i in range(len(all_embeddings)):
    file_name = all_file_names[i]
    cluster_id = labels[i]
    cluster_files[cluster_id].add(file_name)

file_counts = {0: len(cluster_files[0]), 1: len(cluster_files[1])}
print(f"🔮 Файлов в кластере 0: {file_counts[0]}, в кластере 1: {file_counts[1]}")

# Применяем смещение в пользу основного кластера
if file_counts[0] >= file_counts[1] * (1 + MAIN_CLUSTER_BIAS):
    main_cluster = 0
    print(f"🔮 Основной кластер 0 (охват: {file_counts[0]}/{len(wav_files)} файлов, bias: {MAIN_CLUSTER_BIAS})")
else:
    main_cluster = 1
    print(f"🔮 Основной кластер 1 (охват: {file_counts[1]}/{len(wav_files)} файлов, bias: {MAIN_CLUSTER_BIAS})")

# Точное определение ролей через расстояния до центроидов
print("🔮 Точное определение ролей через расстояния до центроидов...")
centroid_0 = kmeans.cluster_centers_[0]
centroid_1 = kmeans.cluster_centers_[1]

speaker_roles = {}
for (file_name, speaker), embedding in file_speaker_avg_embeddings.items():
    # Преобразуем эмбеддинг в PCA-пространство
    emb_pca = pca.transform(embedding.reshape(1, -1))[0]
    
    # Вычисляем расстояния до центроидов
    dist_to_0 = np.linalg.norm(emb_pca - centroid_0)
    dist_to_1 = np.linalg.norm(emb_pca - centroid_1)
    
    # Определяем принадлежность к основному кластеру с порогом сходства
    if main_cluster == 0:
        is_you = dist_to_0 < dist_to_1 * (1 - SIMILARITY_THRESHOLD / 10)
    else:
        is_you = dist_to_1 < dist_to_0 * (1 - SIMILARITY_THRESHOLD / 10)
    
    speaker_roles[(file_name, speaker)] = is_you
    print(f"  🎯 {file_name}:{speaker} -> {'я' if is_you else 'собеседник'} "
          f"(d0={dist_to_0:.2f}, d1={dist_to_1:.2f})")

# Проверка качества диаризации
for audio_path in wav_files:
    if audio_path.name not in file_to_speakers: 
        continue
        
    speakers = file_to_speakers[audio_path.name]
    if len(speakers) < 2: 
        continue
        
    embeddings = []
    for speaker in speakers:
        emb = file_speaker_avg_embeddings.get((audio_path.name, speaker))
        if emb is not None:
            embeddings.append(emb)
    
    if len(embeddings) >= 2:
        avg_sim = average_pairwise_similarity(embeddings)
        # Более чувствительный порог для похожих голосов
        if avg_sim > SIMILARITY_THRESHOLD:
            print(f"⚠️ Внимание! В файле {audio_path.name} голоса слишком похожи (сходство: {avg_sim:.2f}).")

# === Второй проход: транскрипция ===
print("🔍 Второй проход: транскрипция и формирование JSON...")
processed_files = 0

for idx, audio_path in enumerate(wav_files, 1):
    rel_path = audio_path.relative_to(DST)
    output_path = OUTPUT_DIR / rel_path.with_suffix(".json")
    
    if output_path.exists():
        print(f"⏩ ({idx}/{len(wav_files)}) Пропуск (уже обработан): {rel_path}")
        continue
        
    if audio_path not in diarization_data:
        print(f"  ⚠️ ({idx}/{len(wav_files)}) Пропуск (нет данных диаризации): {rel_path}")
        continue
        
    print(f"\n📝 ({idx}/{len(wav_files)}) {rel_path}")
    
    try:
        file_segments = diarization_data[audio_path]
        caller_id = extract_phone_number(str(rel_path))
        
        diarization_segments = []
        for start, end, speaker in file_segments:
            is_you = speaker_roles.get((audio_path.name, speaker), False)
            diarization_segments.append({
                "start": start,
                "end": end,
                "speaker": speaker,
                "is_you": is_you,
                "text": ""  # Будем добавлять целые сегменты
            })
        
        vad_options = VadOptions(
            onset=VAD_ONSET,
            offset=VAD_OFFSET,
            min_speech_duration_ms=MIN_SPEECH_DURATION_MS
        )
        
        # Транскрипция целыми сегментами (без разбивки на слова)
        segments, info = whisper_model.transcribe(
            str(audio_path),
            language=WHISPER_LANGUAGE,
            beam_size=WHISPER_BEAM_SIZE,
            vad_filter=True,
            word_timestamps=False,  # Отключаем поразбивку на слова
            vad_parameters=vad_options,
            suppress_tokens=WHISPER_SUPPRESS_TOKENS  # Отключаем цензуру
        )
        
        # Собираем сегменты транскрипции целиком
        whisper_segments = []
        for segment in segments:
            whisper_segments.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip()
            })
        print(f"  🔠 Распознано сегментов: {len(whisper_segments)}")
        
        # Распределяем целые сегменты транскрипции по сегментам диаризации
        for seg_trans in whisper_segments:
            best_overlap = 0
            best_seg = None
            trans_duration = seg_trans["end"] - seg_trans["start"]
            
            for d_seg in diarization_segments:
                overlap_start = max(seg_trans["start"], d_seg["start"])
                overlap_end = min(seg_trans["end"], d_seg["end"])
                overlap_duration = max(0, overlap_end - overlap_start)
                
                if trans_duration > 0:
                    overlap_ratio = overlap_duration / trans_duration
                else:
                    overlap_ratio = 0
                
                if overlap_ratio > best_overlap:
                    best_overlap = overlap_ratio
                    best_seg = d_seg
            
            if best_seg and best_overlap > 0.3:
                if best_seg["text"]:
                    best_seg["text"] += " " + seg_trans["text"]
                else:
                    best_seg["text"] = seg_trans["text"]
        
        # Фильтруем пустые сегменты
        if SUPPRESS_EMPTY:
            diarization_segments = [s for s in diarization_segments if s["text"].strip()]
        
        # Объединяем короткие сегменты
        enriched_segments = merge_short_segments(diarization_segments)
        
        # Обновление эмбеддингов известных собеседников
        if caller_id and any(not seg["is_you"] for seg in enriched_segments):
            caller_embeddings = []
            for speaker in file_to_speakers[audio_path.name]:
                if not speaker_roles.get((audio_path.name, speaker), True):
                    emb = file_speaker_avg_embeddings.get((audio_path.name, speaker))
                    if emb is not None:
                        caller_embeddings.append(emb)
            
            if caller_embeddings:
                avg_embedding = np.mean(caller_embeddings, axis=0)
                avg_embedding = l2_normalize(avg_embedding).flatten()
                
                if caller_id in known_caller_ids:
                    known_caller_ids[caller_id] = 0.7 * known_caller_ids[caller_id] + 0.3 * avg_embedding
                else:
                    known_caller_ids[caller_id] = avg_embedding
                
                print(f"  🔄 Обновлен эмбеддинг для собеседника: {caller_id}")
        
        caller_id = caller_id or "caller"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(enriched_segments, output_path, rel_path, "0000000000000", caller_id)
        print(f"  💾 Сохранено сегментов: {len(enriched_segments)} → {output_path}")
        processed_files += 1
        
    except Exception as e:
        tb = traceback.extract_tb(e.__traceback__)[0]
        print(f"  ❌ Ошибка при обработке файла: {e}, файл {__file__}, строка {tb.lineno}")

save_known_callers()
print(f"💾 Сохранено {len(known_caller_ids)} известных собеседников")

total_time = format_hhmmss(time.time() - start_all)
print(f"\n✅ Обработка завершена. Обработано файлов: {processed_files}/{len(wav_files)}")
print(f"⏱️ Время выполнения: {total_time}")

print("\n✅ Выполнение process_audio.py завершено.")