#!/usr/bin/env python3
# === Версия ===
print("\n🔢 Версия скрипта process_audio.py 2.60 (Stable GPU Enhanced)")

"""
Требуемые зависимости:

=== Системные пакеты (deb) ===
python3-pip
ffmpeg
dpkg-dev
unzip

=== CUDA/cuDNN (deb) ===
libcudnn9-cuda-12 
libcudnn9-dev-cuda-12
libcublas-12-6 
libcublas-dev-12-6 
cuda-toolkit-12-config-common 
cuda-toolkit-12-6-config-common 
cuda-toolkit-config-common

=== Python-пакеты (WHL) ===
faster-whisper==1.1.0
pyannote-audio==3.3.2
transformers==4.28.1
librosa==0.10.0
hydra-core==1.3.2
faiss-gpu==1.7.2
scikit-learn==1.7.1
soundfile==0.13.1
torch==2.2.2
numpy==1.26.4
scipy==1.13.0
"""

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
from faster_whisper.vad import VadOptions  # Импорт правильного класса VAD
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
            # Используем косинусное сходство (1 - расстояние)
            similarity = 1 - cosine(embeddings[i], embeddings[j])
            total += similarity
            count += 1
            
    return total / count if count > 0 else 0.0

def merge_short_segments(segments, min_duration=0.3, max_gap=0.3):
    """Объединяет короткие сегменты с учетом пауз"""
    if not segments:
        return []
    
    merged = []
    current = segments[0].copy()
    
    for seg in segments[1:]:
        gap = seg["start"] - current["end"]
        same_speaker = current["is_you"] == seg["is_you"]
        
        # Если промежуток маленький, сегмент короткий и спикер тот же - объединяем
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

# Загрузка известных собеседников
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
            "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",  # Нормализация громкости
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

# Словарь для связи файл -> спикеры
file_to_speakers = defaultdict(set)

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
        
        # Диаризация с обработкой ошибок
        try:
            diarization = pipeline(str(audio_path), num_speakers=2)
        except Exception as e:
            print(f"    ⚠️ Ошибка диаризации: {e}. Повторная попытка с num_speakers=2")
            try:
                diarization = pipeline(str(audio_path), num_speakers=2)
            except:
                print(f"    ❌ Повторная ошибка диаризации. Используем fallback подход")
                # Fallback: создаем искусственную диаризацию
                diarization = []
                segment = Segment(0, file_duration)
                diarization.append((segment, "SPEAKER_00"))
        
        # Сохраняем данные диаризации
        file_segments = []
        speaker_embeddings = defaultdict(list)
        
        # Собираем всех спикеров для этого файла
        file_speakers = set()
        
        for segment in diarization.itertracks(yield_label=True):
            if isinstance(segment, tuple) and len(segment) == 3:
                turn, _, speaker = segment
                seg = Segment(turn.start, turn.end)
                
                # Корректируем границы сегмента
                seg = Segment(
                    max(0, min(seg.start, file_duration - 0.01)),
                    min(file_duration, max(seg.end, 0.01))
                )
                
                # Пропускаем слишком короткие или некорректные сегменты (уменьшено до 200 мс)
                if seg.duration < 0.2 or seg.end <= seg.start:
                    continue
                
                file_segments.append((seg.start, seg.end, speaker))
                file_speakers.add(speaker)
                
                # Извлекаем эмбеддинг для сегмента
                try:
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
                    
                    # Если эмбеддинг многомерный - усредняем
                    if embedding.ndim > 1:
                        embedding = np.mean(embedding, axis=0)
                    
                    # Проверяем размерность эмбеддингов
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

# Уменьшаем размерность с помощью PCA
pca = PCA(n_components=min(50, len(embeddings_array)-1))
embeddings_reduced = pca.fit_transform(embeddings_array)

# Кластеризация
kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(embeddings_reduced)
labels = kmeans.labels_

# === Улучшенное определение "я" по охвату файлов ===
print("🔮 Улучшенная идентификация 'я' по охвату файлов...")

# Считаем уникальные файлы для каждого кластера
cluster_files = {0: set(), 1: set()}
for i in range(len(all_embeddings)):
    file_name = all_file_names[i]
    cluster_id = labels[i]
    cluster_files[cluster_id].add(file_name)

# Выбираем кластер с максимальным охватом файлов
file_counts = {0: len(cluster_files[0]), 1: len(cluster_files[1])}
print(f"🔮 Файлов в кластере 0: {file_counts[0]}, в кластере 1: {file_counts[1]}")

if file_counts[0] >= file_counts[1]:
    me_cluster = 0
    print(f"🔮 Кластер 0 идентифицирован как 'я' (охват: {file_counts[0]}/{len(wav_files)} файлов)")
else:
    me_cluster = 1
    print(f"🔮 Кластер 1 идентифицирован как 'я' (охват: {file_counts[1]}/{len(wav_files)} файлов)")

# Проверка качества диаризации в проблемных файлах
for audio_path in wav_files:
    if audio_path.name not in file_to_speakers: 
        continue
        
    speakers = file_to_speakers[audio_path.name]
    if len(speakers) < 2: 
        continue  # Пропускаем файлы с 1 спикером
        
    embeddings = []
    for speaker in speakers:
        # Получаем усредненный эмбеддинг спикера
        emb = next((e for e, f, s in zip(all_embeddings, all_file_names, all_speaker_keys) 
                  if f == audio_path.name and s == speaker), None)
        if emb is not None:
            embeddings.append(emb)
    
    # Проверяем сходство голосов внутри файла
    if len(embeddings) >= 2:
        avg_sim = average_pairwise_similarity(embeddings)
        if avg_sim > 0.85:  # Порог сходства
            print(f"⚠️ Внимание! В файле {audio_path.name} голоса слишком похожи (сходство: {avg_sim:.2f}).")
            print("   Ручная проверка рекомендована.")

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
        
    if audio_path not in diarization_data:
        print(f"  ⚠️ ({idx}/{len(wav_files)}) Пропуск (нет данных диаризации): {rel_path}")
        continue
        
    print(f"\n📝 ({idx}/{len(wav_files)}) {rel_path}")
    
    try:
        file_segments = diarization_data[audio_path]
        caller_id = extract_phone_number(str(rel_path))
        
        # Создаем структуру для обогащенных сегментов
        diarization_segments = []
        for start, end, speaker in file_segments:
            is_you = speaker_roles.get((audio_path.name, speaker), False)
            diarization_segments.append({
                "start": start,
                "end": end,
                "speaker": speaker,
                "is_you": is_you,
                "text_words": []  # Будем собирать слова
            })
        
        # Транскрипция с метками слов (используем правильные параметры VAD)
        vad_options = VadOptions(
            onset=0.35,  # Порог для начала речи (более низкое значение = более чувствительное)
            offset=0.35,  # Порог для окончания речи
            min_speech_duration_ms=150  # Минимальная длительность речи в мс
        )
        
        segments, info = whisper_model.transcribe(
            str(audio_path),
            language="ru",
            beam_size=5,
            vad_filter=True,
            word_timestamps=True,
            vad_parameters=vad_options
        )
        
        # Собираем все слова
        all_words = []
        for segment in segments:
            for word in segment.words:
                all_words.append({
                    "text": word.word,
                    "start": word.start,
                    "end": word.end
                })
        print(f"  🔠 Распознано слов: {len(all_words)}")
        
        # Распределение слов по сегментам диаризации
        for word in all_words:
            best_overlap = 0
            best_seg = None
            
            for d_seg in diarization_segments:
                overlap_start = max(word["start"], d_seg["start"])
                overlap_end = min(word["end"], d_seg["end"])
                overlap_duration = max(0, overlap_end - overlap_start)
                word_duration = word["end"] - word["start"]
                
                if word_duration > 0:
                    overlap_ratio = overlap_duration / word_duration
                    if overlap_ratio > best_overlap:
                        best_overlap = overlap_ratio
                        best_seg = d_seg
            
            if best_seg and best_overlap > 0.3:
                best_seg["text_words"].append(word["text"])
        
        # Формируем текст для сегментов
        for d_seg in diarization_segments:
            if d_seg["text_words"]:
                d_seg["text"] = " ".join(d_seg["text_words"]).strip()
            else:
                d_seg["text"] = ""
        
        # Объединяем короткие сегменты (только одного спикера)
        enriched_segments = merge_short_segments(diarization_segments)
        enriched_segments = [s for s in enriched_segments if s["text"]]
        
        # Проверяем наличие нескольких спикеров
        unique_speakers = len(set(seg["is_you"] for seg in enriched_segments))
        
        # Улучшенная обработка файлов с одним спикером
        if unique_speakers == 1:
            if enriched_segments and enriched_segments[0]["is_you"]:
                print("  ⚠️ Внимание: в файле обнаружен только один спикер, помеченный как 'я'! Проверьте вручную.")
            else:
                print("  ✅ В файле один спикер: помечен как собеседник.")
                for seg in enriched_segments:
                    seg["is_you"] = False
        
        # Сохраняем/обновляем данные о собеседнике
        if caller_id and unique_speakers > 1:
            # Находим эмбеддинг собеседника
            caller_embeddings = []
            for speaker in file_to_speakers[audio_path.name]:
                if not speaker_roles.get((audio_path.name, speaker), True):
                    # Получаем эмбеддинги для этого спикера
                    speaker_embs = [e for i, e in enumerate(all_embeddings) 
                                  if all_file_names[i] == audio_path.name 
                                  and all_speaker_keys[i] == speaker]
                    if speaker_embs:
                        caller_embeddings.extend(speaker_embs)
            
            if caller_embeddings:
                avg_embedding = np.mean(caller_embeddings, axis=0)
                avg_embedding = l2_normalize(avg_embedding).flatten()
                
                # Обновляем или добавляем эмбеддинг
                if caller_id in known_caller_ids:
                    # Экспоненциальное скользящее среднее
                    known_caller_ids[caller_id] = 0.7 * known_caller_ids[caller_id] + 0.3 * avg_embedding
                else:
                    known_caller_ids[caller_id] = avg_embedding
                
                print(f"  🔄 Обновлен эмбеддинг для собеседника: {caller_id}")
        
        # Сохранение результатов
        caller_id = caller_id or "caller"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(enriched_segments, output_path, rel_path, "0000000000000", caller_id)
        print(f"  💾 Сохранено сегментов: {len(enriched_segments)} → {output_path}")
        processed_files += 1
        
    except Exception as e:
        tb = traceback.extract_tb(e.__traceback__)[0]
        print(f"  ❌ Ошибка при обработке файла: {e}, файл {__file__}, строка {tb.lineno}")

# Сохраняем известных собеседников
save_known_callers()
print(f"💾 Сохранено {len(known_caller_ids)} известных собеседников")

total_time = format_hhmmss(time.time() - start_all)
print(f"\n✅ Обработка завершена. Обработано файлов: {processed_files}/{len(wav_files)}")
print(f"⏱️ Время выполнения: {total_time}")

# === Завершение ===
print("\n✅ Выполнение process_audio.py завершено.")