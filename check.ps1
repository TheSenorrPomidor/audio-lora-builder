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





	Write-Host "❌ СТОП ТЕСТ"; exit 1
<#




#>





