#!/usr/bin/env python3
# === Версия ===
print("\n🔢 Версия скрипта process_audio.py 2.85 (Advanced Voice Vector Matching)")

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
import pickle
from scipy.spatial.distance import cosine

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

def merge_short_segments(segments, min_duration=0.5, max_gap=0.5):
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

# === Глобальный эталонный эмбеддинг для "я" ===
YOU_EMBEDDING_FILE = Path("/root/audio-lora-builder/config/you_embedding.pkl")
you_embedding = None  # Глобальный эталонный эмбеддинг для вашего голоса

if YOU_EMBEDDING_FILE.exists():
    try:
        with open(YOU_EMBEDDING_FILE, "rb") as f:
            you_embedding = pickle.load(f)
        print(f"🔊 Загружен эталонный эмбеддинг для 'я'")
    except:
        print("⚠️ Не удалось загрузить эталонный эмбеддинг для 'я'")

def save_you_embedding():
    if you_embedding is not None:
        with open(YOU_EMBEDDING_FILE, "wb") as f:
            pickle.dump(you_embedding, f)

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

# === Глобальный словарь системных голосов ===
SYSTEM_VOICES_FILE = Path("/root/audio-lora-builder/config/system_voices.pkl")
system_voices = []  # список эмбеддингов системных голосов

if SYSTEM_VOICES_FILE.exists():
    try:
        with open(SYSTEM_VOICES_FILE, "rb") as f:
            system_voices = pickle.load(f)
        print(f"🔊 Загружено {len(system_voices)} системных голосов")
    except:
        print("⚠️ Не удалось загрузить системные голоса")

def save_system_voices():
    with open(SYSTEM_VOICES_FILE, "wb") as f:
        pickle.dump(system_voices, f)

def is_system_voice(embedding, threshold=0.85):
    """Определяет, является ли голос системным"""
    if not system_voices:
        return False
        
    for sys_emb in system_voices:
        similarity = 1 - cosine(embedding, sys_emb)
        if similarity > threshold:
            return True
    return False

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
file_speaker_embeddings = defaultdict(list)  # (file_name, speaker) -> [embeddings]
file_to_speakers = defaultdict(set)
diarization_data = {}

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
        file_speakers = set()
        
        for segment in diarization.itertracks(yield_label=True):
            if isinstance(segment, tuple) and len(segment) == 3:
                turn, _, speaker = segment
                seg = Segment(turn.start, turn.end)
                
                seg = Segment(
                    max(0, min(seg.start, file_duration - 0.01)),
                    min(file_duration, max(seg.end, 0.01))
                )
                
                if seg.duration < 0.2 or seg.end <= seg.start:
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
                        tb = traceback.extract_tb(e.__traceback__)[0]
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
                    
                    # Сохраняем эмбеддинг для каждого сегмента
                    file_speaker_embeddings[(audio_path.name, speaker)].append(embedding)
                except Exception as e:
                    tb = traceback.extract_tb(e.__traceback__)[0]
                    print(f"    ⚠️ Ошибка при обработке сегмента: {e}, файл {__file__}, строка {tb.lineno}")
                    continue
        
        diarization_data[audio_path] = file_segments
        file_to_speakers[audio_path.name] = file_speakers
            
    except Exception as e:
        tb = traceback.extract_tb(e.__traceback__)[0]
        print(f"  ❌ Ошибка при извлечении эмбеддингов: {e}, файл {__file__}, строка {tb.lineno}")

# === Второй проход: транскрипция и определение спикеров ===
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
        caller_id = extract_phone_number(str(rel_path)) or "caller"
        
        # Собираем все эмбеддинги для этого файла
        speaker_embeddings = {}
        for speaker in file_to_speakers[audio_path.name]:
            embeddings_list = file_speaker_embeddings.get((audio_path.name, speaker), [])
            if embeddings_list:
                # Усредняем эмбеддинги для каждого спикера
                avg_embedding = np.mean(embeddings_list, axis=0)
                speaker_embeddings[speaker] = avg_embedding
        
        diarization_segments = []
        you_detected = False
        
        for start, end, speaker in file_segments:
            # Получаем эмбеддинг для текущего спикера
            current_embedding = speaker_embeddings.get(speaker)
            
            # Инициализация флага
            is_you = False
            
            # Если есть эталонный эмбеддинг "я"
            if you_embedding is not None and current_embedding is not None:
                # Вычисляем косинусное сходство
                similarity = 1 - cosine(you_embedding, current_embedding)
                
                # Определяем порог динамически
                threshold = 0.75  # Базовый порог
                
                # Если сходство высокое, помечаем как "я"
                if similarity > threshold:
                    is_you = True
                    you_detected = True
                    print(f"  🎯 Сходство с 'я': {similarity:.2f} (порог: {threshold:.2f})")
            
            # Если "я" еще не определено, но есть эмбеддинг
            elif current_embedding is not None:
                # Если это первый файл, инициализируем эталон
                if you_embedding is None:
                    you_embedding = current_embedding
                    is_you = True
                    you_detected = True
                    print("  🎯 Инициализация эталонного эмбеддинга 'я'")
            
            # Проверка на системный голос
            if current_embedding is not None and is_system_voice(current_embedding):
                is_you = False
                print(f"  🤖 Обнаружен системный голос: {speaker}")
            
            diarization_segments.append({
                "start": start,
                "end": end,
                "speaker": speaker,
                "is_you": is_you,
                "text": ""
            })
        
        # Обновление эталонного эмбеддинга "я"
        if you_detected:
            # Находим лучший эмбеддинг "я" в этом файле
            best_you_embedding = None
            max_similarity = -1
            
            for speaker, embedding in speaker_embeddings.items():
                if embedding is not None and any(seg["is_you"] and seg["speaker"] == speaker for seg in diarization_segments):
                    if you_embedding is not None:
                        similarity = 1 - cosine(you_embedding, embedding)
                        if similarity > max_similarity:
                            max_similarity = similarity
                            best_you_embedding = embedding
            
            # Если нашли подходящий эмбеддинг, обновляем эталон
            if best_you_embedding is not None:
                # Взвешенное обновление (80% старый + 20% новый)
                if you_embedding is None:
                    you_embedding = best_you_embedding
                else:
                    you_embedding = 0.8 * you_embedding + 0.2 * best_you_embedding
                save_you_embedding()
                print(f"  🔄 Обновлен эталонный эмбеддинг 'я'")
        
        vad_options = VadOptions(
            onset=0.35,
            offset=0.35,
            min_speech_duration_ms=150
        )
        
        # Транскрипция целыми сегментами
        segments, info = whisper_model.transcribe(
            str(audio_path),
            language="ru",
            beam_size=5,
            vad_filter=True,
            word_timestamps=False,
            vad_parameters=vad_options
        )
        
        # Собираем сегменты транскрипции
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
        diarization_segments = [s for s in diarization_segments if s["text"].strip()]
        
        # Обнаружение системных сообщений по тексту
        robotic_phrases = [
            "звониваться", "оставайтесь на линии", "ожидайте", "соединяем",
            "пожалуйста не вешайте трубку", "добро пожаловать", "ваш звонок очень важен"
        ]
        
        for seg in diarization_segments:
            if any(phrase in seg["text"].lower() for phrase in robotic_phrases):
                seg["is_you"] = False
                # Сохраняем эмбеддинг для будущего использования
                speaker_embed = speaker_embeddings.get(seg["speaker"])
                if speaker_embed is not None:
                    system_voices.append(speaker_embed)
                    print(f"  🤖 Обнаружено системное сообщение: {seg['text'][:50]}...")
        
        # Объединяем короткие сегменты
        enriched_segments = merge_short_segments(diarization_segments)
        
        # Обновляем эмбеддинги собеседника
        if caller_id != "caller" and any(not seg["is_you"] for seg in enriched_segments):
            caller_embeddings = []
            for speaker, embeddings_list in file_speaker_embeddings.items():
                if speaker[0] == audio_path.name:
                    if not any(seg["is_you"] for seg in enriched_segments if seg["speaker"] == speaker[1]):
                        caller_embeddings.extend(embeddings_list)
            
            if caller_embeddings:
                avg_embedding = np.mean(caller_embeddings, axis=0)
                avg_embedding = l2_normalize(avg_embedding).flatten()
                
                if caller_id in known_caller_ids:
                    # Взвешенное обновление эмбеддинга
                    known_caller_ids[caller_id] = 0.7 * known_caller_ids[caller_id] + 0.3 * avg_embedding
                else:
                    known_caller_ids[caller_id] = avg_embedding
                
                print(f"  🔄 Обновлен эмбеддинг для собеседника: {caller_id}")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(enriched_segments, output_path, rel_path, "0000000000000", caller_id)
        print(f"  💾 Сохранено сегментов: {len(enriched_segments)} → {output_path}")
        processed_files += 1
        
    except Exception as e:
        tb = traceback.extract_tb(e.__traceback__)[0]
        print(f"  ❌ Ошибка при обработке файла: {e}, файл {__file__}, строка {tb.lineno}")

save_known_callers()
save_system_voices()
save_you_embedding()
print(f"💾 Сохранено {len(known_caller_ids)} известных собеседников, {len(system_voices)} системных голосов и эталон 'я'")

total_time = format_hhmmss(time.time() - start_all)
print(f"\n✅ Обработка завершена. Обработано файлов: {processed_files}/{len(wav_files)}")
print(f"⏱️ Время выполнения: {total_time}")

print("\n✅ Выполнение process_audio.py завершено.")