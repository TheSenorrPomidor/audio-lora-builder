$DistroName = "audio-lora"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$KeyDir = Join-Path $ScriptDir "ssh_keys"
$TempDir = Join-Path $ScriptDir "temp"
$RootfsDir = Join-Path $ScriptDir "rootfs"
$FinalRootfs = Join-Path $RootfsDir "audio_lora_rootfs.tar.gz"
$BaseRootfs = Join-Path $RootfsDir "Ubuntu_2204.1.7.0_x64_rootfs.tar.gz"
$BundleZipFile = Join-Path $TempDir "Ubuntu2204AppxBundle.zip"
$BundleExtractPath = Join-Path $TempDir "Ubuntu2204AppxBundle"
# Whisper-механизм: faster-whisper или whisperx
$WhisperImpl = "faster-whisper"
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
# === Функция поиска пакетов WHL в локальной папке ===
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
		$name = $name.ToLower() -replace '[._]+', '-'
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
# === Функция получения/чтения HuggingFaceToken из/в файл huggingface_hub_token.txt ===
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










<#

	Write-Host "❌ СТОП ТЕСТ"; exit 1
	
	
	
The cache for model files in Transformers v4.22.0 has been updated. Migrating your old cache. This is a one-time only operation. You can interrupt this and resume the migration later on by calling `transformers.utils.move_cache()`.
0it [00:00, ?it/s]

#>





