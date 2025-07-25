$DistroName = "audio-lora"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$TempDir = Join-Path $ScriptDir "temp"
$RootfsDir = Join-Path $ScriptDir "rootfs"
$FinalRootfs = Join-Path $RootfsDir "audio_lora_rootfs.tar.gz"
$BaseRootfs = Join-Path $RootfsDir "Ubuntu_2204.1.7.0_x64_rootfs.tar.gz"
$BundleZipFile = Join-Path $TempDir "Ubuntu2204AppxBundle.zip"
$BundleExtractPath = Join-Path $TempDir "Ubuntu2204AppxBundle"
# Whisper-механизм: faster-whisper или whisperx
$WhisperImpl = "whisperx"
#wsl --export audio-lora "D:\VM\WSL2\audio-lora-builder\rootfs\audio_lora_rootfs.tar.gz"
#sudo rm -rf /root/.cache/torch/pyannote



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
function Get-WhlInventory($WhlDir, $DistroName) {
    $Inventory = @{}
	if (-not (Test-Path $WhlDir)) { New-Item -ItemType Directory -Path $WhlDir | Out-Null }
    $WhlFiles = Get-ChildItem -Path $WhlDir -Filter *.whl

	#Счетчик для процента прогресса
	$step = 1
	$total = $WhlFiles.Count
	
    foreach ($whl in $WhlFiles) {
		
			#Write-Host $name=$version
			#Процент прогресса
			$percent = [math]::Round(($step / $total) * 100)
			Write-Host -NoNewline "`r📦 Чтение .whl файлов: $percent%"
			$step++                                                     



        # Преобразуем путь в WSL-совместимый 
		$whlPathWsl = Convert-WindowsPathToWsl $whl.FullName

        # Извлекаем Name и Version из METADATA файла
		$cmd = "unzip -p '$whlPathWsl' '*.dist-info/METADATA' | grep -E '^(Name|Version):'"
        $meta = wsl -d $DistroName -- bash -c $cmd

        # Разбор результата
        $lines = $meta -split "`n"
       # $name = ($lines | Where-Object { $_ -like 'Name:*' }) -replace 'Name:\s*', ''
       # $version = ($lines | Where-Object { $_ -like 'Version:*' }) -replace 'Version:\s*', ''

		$name = ($lines | Where-Object { $_ -like 'Name:*' } | Select-Object -First 1) -replace 'Name:\s*', ''
		$name = $name.ToLower()
		$version = ($lines | Where-Object { $_ -like 'Version:*' } | Select-Object -First 1) -replace 'Version:\s*', ''

		
        if ($name -and $version) {
            $Inventory["$name==$version"] = $whl.FullName
        }
    }

	Write-Host "`r📦 Чтение .whl файлов завершено.           "

	return $Inventory.GetEnumerator() | ForEach-Object {
	@{
		Name    = ($_).Key -split '==' | Select-Object -First 1
		Version = ($_).Key -split '==' | Select-Object -Skip 1
		Path    = ($_).Value
	}
	}
}





wsl -d $DistroName -- bash -c "echo 'nameserver 8.8.8.8' > /etc/resolv.conf"
###############################################################################
###############################################################################
###############################################################################


Write-Host "`n5.7 📦 Расширенная предзагрузка моделей pyannote..."

$PreloadPath = Join-Path $ScriptDir "temp\preload_diarization_models.py"

@"
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
fake_waveform = torch.zeros(1, 16000)
pipeline({"waveform": fake_waveform, "sample_rate": 16000})

# Загружаем реальный mp3-файл
audio_path = "/mnt/d/VM/WSL2/audio-lora-builder/audio_src/Сашенька!(0079211058204)_20250622221226.mp3"
output_path = "/mnt/d/VM/WSL2/audio-lora-builder/audio_src/Сашенька!(0079211058204)_20250622221226.vad.rttm"

audio = Audio(sample_rate=16000)
waveform, sample_rate = audio(audio_path)

# Диаризация
diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate}, num_speakers=2)
#diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate})

# Получаем только VAD-подобные участки (спикеры != <NA>)
vad_timeline = diarization.get_timeline()

# Сохраняем RTTM
with open(output_path, "w") as f:
    for turn, track, speaker in diarization.itertracks(yield_label=True):
        start = turn.start
        duration = turn.end - turn.start
        f.write(f"SPEAKER test 1 {start:.3f} {duration:.3f} <NA> <NA> {speaker} <NA> <NA>\n")
"@ | Set-Content -Encoding UTF8 -Path $PreloadPath

$PreloadPathWsl = Convert-WindowsPathToWsl $PreloadPath
wsl -d $DistroName -- bash -c "python3 '$PreloadPathWsl'"

Write-Host "✅ Кэш моделей pyannote полностью загружен, проверен на файле Саши с VAD-фильтрацией."




	Write-Host "❌ СТОП ТЕСТ"; exit 1
<#



#>





