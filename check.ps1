$DistroName = "audio-lora"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$TempDir = Join-Path $ScriptDir "temp"
$RootfsDir = Join-Path $ScriptDir "rootfs"
$FinalRootfs = Join-Path $RootfsDir "audio_lora_rootfs.tar.gz"
$BaseRootfs = Join-Path $RootfsDir "Ubuntu_2204.1.7.0_x64_rootfs.tar.gz"
$BundleZipFile = Join-Path $TempDir "Ubuntu2204AppxBundle.zip"
$BundleExtractPath = Join-Path $TempDir "Ubuntu2204AppxBundle"
# Whisper-механизм: faster-whisper или whisperx
$WhisperImpl = "faster-whisper"
#wsl --export audio-lora "D:\VM\WSL2\audio-lora-builder\rootfs\audio_lora_rootfs.tar.gz"
#sudo rm -rf /root/.cache/torch/pyannote



#

$KeyDir = Join-Path $ScriptDir "ssh_keys"



#












# === Функции ===
# === Функция преобразования путей Windows в WSL ===
function Convert-WindowsPathToWsl {
    param (
        [Parameter(Mandatory=$true)]
        [string]$WindowsPath
    )
    $WslPath = $WindowsPath -replace '\\', '/' -replace '^([A-Za-z]):', '/mnt/$1'
    return $WslPath.ToLower()
}

function Get-HuggingFaceToken {
    $TokenPath = Join-Path $KeyDir "huggingface_hub_token.txt"

    if (-not (Test-Path $TokenPath)) {
        Write-Host "🔐 Токен не найден. Введите токен Hugging Face:"
        $Token = Read-Host "HuggingFace Token"
        $Token = $Token.Trim()

        try {
            $Headers = @{ Authorization = "Bearer $Token" }
            $Parsed = Invoke-RestMethod -Uri "https://huggingface.co/api/whoami-v2" -Headers $Headers -Method Get
            if (-not $Parsed.name) {
                throw "Токен не прошёл верификацию"
            }

            Write-Host "✅ Авторизация успешна. Учётная запись: $($Parsed.name) <$($Parsed.email)>"

            New-Item -Path $KeyDir -ItemType Directory -Force | Out-Null
            $Token | Set-Content -Encoding UTF8 -Path $TokenPath
        }
        catch {
            Write-Host "❌ Ошибка проверки токена:"
            Write-Host $_.Exception.Message
            Write-Host "⛔ Завершение."
            exit 1
        }

    }

    return (Get-Content $TokenPath -Raw).Trim()
}






wsl -d $DistroName -- bash -c "echo 'nameserver 8.8.8.8' > /etc/resolv.conf"
###############################################################################
###############################################################################
###############################################################################
#НАДО КОПИРОВАТЬ ПАПКУ PYANNOTE В \\wsl.localhost\audio-lora\root\.cache\torch

Write-Host "`n5.6 📦 Расширенная предзагрузка моделей pyannote..."
$ModelCacheWin = Join-Path $TempDir "huggingface\torch\pyannote"
$ModelCacheWinWsl = Convert-WindowsPathToWsl $ModelCacheWin
$ModelCacheLocalWsl = "/root/.cache/torch/pyannote"
$TempPreloadPyFile = Join-Path $TempDir "preload_diarization_models.py"
$TempPreloadPyFileWsl = Convert-WindowsPathToWsl $TempPreloadPyFile	
$PyannoteCacheSearchPattern = "*pyannote*"
$PythonLoadScript = @"
from pyannote.audio import Model
from pyannote.audio.pipelines import SpeakerDiarization
from pyannote.audio.core.io import Audio
from pyannote.core import Segment
import torch
import os


import warnings
warnings.filterwarnings("ignore")  # отключает все предупреждения
import logging
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)


# Прогрев моделей
Model.from_pretrained("pyannote/segmentation", use_auth_token=True)
Model.from_pretrained("pyannote/embedding", use_auth_token=True)

# Загружаем диаризатор
pipeline = SpeakerDiarization.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=True)

# Настраиваем чувствительность
pipeline.onset = 0.767                 # 0.767 по умолчанию. Порог включения речи — насколько сильно модель должна "поверить", что начался голос. меньше значение → больше чувствительность
pipeline.offset = 0.377                # 0.377 по умолчанию. Порог выключения речи — насколько сильно модель должна "поверить", что речь закончилась. Меньше значение → чувствительнее к тишине, будет быстрее обрывать речь.
pipeline.min_duration_on = 0.136       # 0.136 по умолчанию. Минимальная продолжительность речи, чтобы она была учтена как отдельный фрагмент. Если голос прозвучал менее чем на 136 мс, он игнорируется полностью — считается шумом (например, шорох, вздох, "эм").
pipeline.min_duration_off = 0.067      # 0.067 по умолчанию Минимальная продолжительность паузы, чтобы считалась настоящей тишиной между спикерами. Если тишина короче 67 мс, она игнорируется и две реплики сливаются в одну.


# Прогрев на фиктивных данных
fake_waveform = torch.zeros(1, 16000 * 5)  # 5 секунд тишины
pipeline({"waveform": fake_waveform, "sample_rate": 16000}, num_speakers=2)
"@ 


#Проверка установленного кэша модели в WSL)
Write-Host "📦 Проверка установленного кэша модели Pyannote в WSL..."
$CheckModelCmd = "bash -c 'ls -1 " + $ModelCacheLocalWsl + "/" + $PyannoteCacheSearchPattern + " 1>/dev/null 2>&1'"


wsl -d $DistroName -- bash -c "$CheckModelCmd"
$ModelCached = $LASTEXITCODE

if ($ModelCached -eq 0) {
	Write-Host "✅ Кэш Pyannote уже установлен на WSL. Пропускаем загрузку."
}
else {
	#Проверяем кэш в temp на Windows
	Write-Host "📦 Кэш в WSL не найден, ищем кэш в temp: $ModelCacheWin"
	$CheckModelCmd = "bash -c 'ls -1 " + $ModelCacheWinWsl + "/" + $PyannoteCacheSearchPattern + " 1>/dev/null 2>&1'"


	wsl -d $DistroName -- bash -c "$CheckModelCmd"
	$ModelDownloaded = $LASTEXITCODE
	
	if ($ModelDownloaded -eq 0) {
		Write-Host "📦 Кэш модели Pyannote найден. Копируем в WSL..."
		wsl -d $DistroName -- bash -c "mkdir -p '$ModelCacheLocalWsl' && cp -r '$ModelCacheWinWsl/'* '$ModelCacheLocalWsl/'"

		Write-Host "✅ Кэш успешно скопирован Windows => WSL."
	}
	else {
		Write-Host "📦 Кэш модели Pyannote не найден. Скачиваем модель с huggingface"

		$PythonLoadScript | Out-File -FilePath $TempPreloadPyFile -Encoding UTF8

		wsl -d $DistroName -- bash -c "python3 '$TempPreloadPyFileWsl'"
		Remove-Item $TempPreloadPyFile -Force

		Write-Host "📦 Кэш модели Pyannote скачен. Копируем кэш WSL → Windows для будущей оффлайн установки..."
		wsl -d $DistroName -- bash -c "mkdir -p '$ModelCacheWinWsl' && cp -r '$ModelCacheLocalWsl/'* '$ModelCacheWinWsl/'"

		Write-Host "✅ Кэш скачен и скопирован Windows => WSL."
	}
}

$Token = Get-HuggingFaceToken




	Write-Host "❌ СТОП ТЕСТ"; exit 1
<#

Поиск токена в ssh_keys;

Чтение, либо интерактивный ввод;

Верификацию токена через WSL + curl; - ТОЛЬКО ЕСЛИ БЫЛ ИНТЕРРАКТИВНЫЙ ВВОД

Вывод имени и почты учётки; - ТОЛЬКО ЕСЛИ БЫЛ ИНТЕРРАКТИВНЫЙ ВВОД

Сохранение токена при успешной проверке. - ТОЛЬКО ЕСЛИ БЫЛ ИНТЕРРАКТИВНЫЙ ВВОД

Возврат токена из файла в папке ssh_keys

#>





