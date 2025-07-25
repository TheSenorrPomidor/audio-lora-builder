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

# Прогрев моделей
Model.from_pretrained("pyannote/segmentation", use_auth_token=True)
Model.from_pretrained("pyannote/embedding", use_auth_token=True)

# Загружаем диаризатор
pipeline = SpeakerDiarization.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=True)

# Прогрев на фиктивных данных
fake_waveform = torch.zeros(1, 16000)
pipeline({"waveform": fake_waveform, "sample_rate": 16000})

# Загружаем реальный mp3-файл
audio_path = "/mnt/d/VM/WSL2/audio-lora-builder/audio_src/Сашенька!(0079211058204)_20250622221226.mp3"
output_path = "/mnt/d/VM/WSL2/audio-lora-builder/audio_src/Сашенька!(0079211058204)_20250622221226.vad.rttm"

audio = Audio(sample_rate=16000)
waveform, sample_rate = audio(audio_path)

# Диаризация
diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate})

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

У меня два вопроса:
1. Файл диаризации наконец создан. Вот файл ```SPEAKER test 1 0.031 0.996 <NA> <NA> SPEAKER_03 <NA> <NA>
SPEAKER test 1 1.381 0.574 <NA> <NA> SPEAKER_03 <NA> <NA>
SPEAKER test 1 2.157 0.928 <NA> <NA> SPEAKER_03 <NA> <NA>
SPEAKER test 1 2.393 1.991 <NA> <NA> SPEAKER_00 <NA> <NA>
SPEAKER test 1 4.148 2.346 <NA> <NA> SPEAKER_03 <NA> <NA>
SPEAKER test 1 6.781 0.894 <NA> <NA> SPEAKER_03 <NA> <NA>
SPEAKER test 1 6.865 2.481 <NA> <NA> SPEAKER_00 <NA> <NA>
SPEAKER test 1 9.802 5.265 <NA> <NA> SPEAKER_03 <NA> <NA>
SPEAKER test 1 16.147 1.586 <NA> <NA> SPEAKER_01 <NA> <NA>
SPEAKER test 1 18.087 0.877 <NA> <NA> SPEAKER_01 <NA> <NA>
SPEAKER test 1 18.965 0.793 <NA> <NA> SPEAKER_02 <NA> <NA>
SPEAKER test 1 20.517 0.489 <NA> <NA> SPEAKER_01 <NA> <NA>
SPEAKER test 1 21.580 0.911 <NA> <NA> SPEAKER_01 <NA> <NA>```
2. Лог запуска здоровый куча всего написана, не понимаю нужно мне это или нет, но лог явно не читаемый для меня, что делать?
Вот лог ```5.7 📦 Расширенная предзагрузка моделей pyannote...
Lightning automatically upgraded your loaded checkpoint from v1.5.4 to v2.5.2. To apply the upgrade to your files permanently, run `python -m pytorch_lightning.utilities.upgrade_checkpoint ../../../../../root/.cache/torch/pyannote/models--pyannote--segmentation/snapshots/660b9e20307a2b0cdb400d0f80aadc04a701fc54/pytorch_model.bin`
Model was trained with pyannote.audio 0.0.1, yours is 3.3.2. Bad things might happen unless you revert pyannote.audio to 0.x.
Model was trained with torch 1.10.0+cu102, yours is 2.7.1+cu118. Bad things might happen unless you revert torch to 1.x.
/usr/local/lib/python3.10/dist-packages/pytorch_lightning/utilities/migration/migration.py:208: You have multiple `ModelCheckpoint` callback states in this checkpoint, but we found state keys that would end up colliding with each other after an upgrade, which means we can't differentiate which of your checkpoint callbacks needs which states. At least one of your `ModelCheckpoint` callbacks will not be able to reload the state.
Lightning automatically upgraded your loaded checkpoint from v1.2.7 to v2.5.2. To apply the upgrade to your files permanently, run `python -m pytorch_lightning.utilities.upgrade_checkpoint ../../../../../root/.cache/torch/pyannote/models--pyannote--embedding/snapshots/4db4899737a38b2d618bbd74350915aa10293cb2/pytorch_model.bin`
Model was trained with pyannote.audio 0.0.1, yours is 3.3.2. Bad things might happen unless you revert pyannote.audio to 0.x.
Model was trained with torch 1.8.1+cu102, yours is 2.7.1+cu118. Bad things might happen unless you revert torch to 1.x.
/usr/local/lib/python3.10/dist-packages/pyannote/audio/core/model.py:692: UserWarning: Model has been trained with a task-dependent loss function. Set 'strict' to False to load the model without its loss function and prevent this warning from appearing.
  warnings.warn(msg)
Lightning automatically upgraded your loaded checkpoint from v1.2.7 to v2.5.2. To apply the upgrade to your files permanently, run `python -m pytorch_lightning.utilities.upgrade_checkpoint ../../../../../root/.cache/torch/pyannote/models--pyannote--embedding/snapshots/4db4899737a38b2d618bbd74350915aa10293cb2/pytorch_model.bin`
Model was trained with pyannote.audio 0.0.1, yours is 3.3.2. Bad things might happen unless you revert pyannote.audio to 0.x.
Model was trained with torch 1.8.1+cu102, yours is 2.7.1+cu118. Bad things might happen unless you revert torch to 1.x.
/usr/local/lib/python3.10/dist-packages/pytorch_lightning/core/saving.py:195: Found keys that are not in the model state dict but in the checkpoint: ['loss_func.W']
/usr/local/lib/python3.10/dist-packages/pyannote/audio/models/blocks/pooling.py:104: UserWarning: std(): degrees of freedom is <= 0. Correction should be strictly less than the reduction factor (input numel divided by output numel). (Triggered internally at /pytorch/aten/src/ATen/native/ReduceOps.cpp:1839.)
  std = sequences.std(dim=-1, correction=1)
✅ Кэш моделей pyannote полностью загружен, проверен на файле Саши с VAD-фильтрацией.```

#>





