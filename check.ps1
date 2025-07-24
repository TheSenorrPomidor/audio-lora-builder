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
from pyannote.audio import Model, Audio
from pyannote.audio.pipelines import SpeakerDiarization, VoiceActivityDetection
import torchaudio
import torch

# Принудительно загружаем модели
Model.from_pretrained("pyannote/segmentation", use_auth_token=True)
Model.from_pretrained("pyannote/embedding", use_auth_token=True)

# Прогреваем pipeline
pipeline = SpeakerDiarization.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=True)
pipeline({"waveform": torch.zeros(1, 16000), "sample_rate": 16000})

# Подгружаем VAD на базе segmentation
vad = VoiceActivityDetection(segmentation="pyannote/segmentation", use_auth_token=True)
audio = Audio(sample_rate=16000)

# Загружаем тестовый mp3
mp3_path = "/mnt/d/VM/WSL2/audio-lora-builder/audio_src/Сашенька!(0079211058204)_20250622221226.mp3"
waveform, sample_rate = torchaudio.load(mp3_path)

# Получаем интервалы речи
speech_timeline = vad({'waveform': waveform, 'sample_rate': sample_rate}).get_timeline()

# Вырезаем только участки с речью
segments = audio.crop({'waveform': waveform, 'sample_rate': sample_rate}, speech_timeline)

# Объединяем вырезанные куски в один аудио-тензор
speech_only = torch.cat([s for s in segments], dim=1)

# Диаризация только по речи
diarization = pipeline({"waveform": speech_only, "sample_rate": sample_rate})

# Сохраняем результат
out_path = "/mnt/d/VM/WSL2/audio-lora-builder/output/test_sasha.rttm"
with open(out_path, "w") as f:
    diarization.write_rttm(f)
"@ | Set-Content -Encoding UTF8 -Path $PreloadPath

$PreloadPathWsl = Convert-WindowsPathToWsl $PreloadPath
wsl -d $DistroName -- bash -c "python3 '$PreloadPathWsl'"

Write-Host "✅ Кэш моделей pyannote полностью загружен, проверен на файле Саши с VAD-фильтрацией."




	Write-Host "❌ СТОП ТЕСТ"; exit 1
<#

ПРОГРЕВ НА ПУСТОМ ФАЙЛЕ
Write-Host "`n5.7 📦 Расширенная предзагрузка моделей pyannote..."

$PreloadPath = Join-Path $ScriptDir "temp\preload_diarization_models.py"

@"
from pyannote.audio import Model
from pyannote.audio.pipelines import SpeakerDiarization
import torch

# Принудительно загружаем модели в кэш
Model.from_pretrained("pyannote/segmentation", use_auth_token=True)
Model.from_pretrained("pyannote/embedding", use_auth_token=True)

# Также прогреваем сам пайплайн
pipeline = SpeakerDiarization.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=True)
fake_waveform = torch.zeros(1, 16000)
pipeline({"waveform": fake_waveform, "sample_rate": 16000})
"@ | Set-Content -Encoding UTF8 -Path $PreloadPath

$PreloadPathWsl = Convert-WindowsPathToWsl $PreloadPath
wsl -d $DistroName -- bash -c "python3 '$PreloadPathWsl'"

Write-Host "✅ Кэш моделей pyannote полностью загружен и готов к оффлайн-использованию."




По поводу несуществующей Voice activity detection, вот карточка доступная мне по ссылке https://huggingface.co/pyannote/voice-activity-detection. Существует же, не?

Model card
Files
xet
Gated model
You have been granted access to this model

Using this open-source model in production?
Consider switching to pyannoteAI for better and faster options.

🎹 Voice activity detection
Relies on pyannote.audio 2.1: see installation instructions.

# 1. visit hf.co/pyannote/segmentation and accept user conditions
# 2. visit hf.co/settings/tokens to create an access token
# 3. instantiate pretrained voice activity detection pipeline

from pyannote.audio import Pipeline
pipeline = Pipeline.from_pretrained("pyannote/voice-activity-detection",
                                    use_auth_token="ACCESS_TOKEN_GOES_HERE")
output = pipeline("audio.wav")

for speech in output.get_timeline().support():
    # active speech between speech.start and speech.end
    ...

Citation
@inproceedings{Bredin2021,
  Title = {{End-to-end speaker segmentation for overlap-aware resegmentation}},
  Author = {{Bredin}, Herv{\'e} and {Laurent}, Antoine},
  Booktitle = {Proc. Interspeech 2021},
  Address = {Brno, Czech Republic},
  Month = {August},
  Year = {2021},
}

@inproceedings{Bredin2020,
  Title = {{pyannote.audio: neural building blocks for speaker diarization}},
  Author = {{Bredin}, Herv{\'e} and {Yin}, Ruiqing and {Coria}, Juan Manuel and {Gelly}, Gregory and {Korshunov}, Pavel and {Lavechin}, Marvin and {Fustes}, Diego and {Titeux}, Hadrien and {Bouaziz}, Wassim and {Gill}, Marie-Philippe},
  Booktitle = {ICASSP 2020, IEEE International Conference on Acoustics, Speech, and Signal Processing},
  Address = {Barcelona, Spain},
  Month = {May},
  Year = {2020},
}


#>





