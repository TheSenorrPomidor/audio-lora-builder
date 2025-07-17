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




# === 5.4 Установка Python-библиотек (faster-whisper или torch + whisperx) ===
Write-Host "`n5.4 📦 Установка Python-библиотек (torch, $WhisperImpl) из temp\pip (Windows)"
$PipCacheWin = Join-Path $TempDir "pip"
$PipCacheWsl = Convert-WindowsPathToWsl $PipCacheWin

$PyWheels = @(
  @{ Name = "whisperx==3.3.1";      Source = "torch"; Impl = "whisperx" },
  @{ Name = "transformers==4.28.1"; Source = "torch"; Impl = "whisperx" },
  @{ Name = "librosa==0.10.0";      Source = "torch"; Impl = "whisperx" },
  @{ Name = "faster-whisper";       Source = "pypi";  Impl = "faster-whisper" }

)
#Проверить какие пакеты .whl установлены в WSL
$PyWheelsMissing = @()
foreach ($pkg in $PyWheels) {
    $IsForThisImpl = ($pkg.Impl -eq "all" -or $pkg.Impl -eq $WhisperImpl)
    if ($IsForThisImpl) {

		$DepName = $pkg.Name -split '==|\+' | Select-Object -First 1
		$DepVersion = $pkg.Name -split '==|\+' | Select-Object -Skip 1 | Select-Object -First 1
		$IsInstalled = wsl -d $DistroName -- bash -c "pip show $DepName > /dev/null && echo ok"
		if ($IsInstalled -eq "ok") {
			Write-Host "✅ $($pkg.Name) уже установлен — пропускаем"
		} else {
			Write-Host "⬇️ $($pkg.Name) не установлен — добавляем в обработку"
			$PyWheelsMissing += $pkg
		}
	}
}

# Проверяем наличие ранее скаченных пакетов .whl в temp/pip
#$PyWheelsToDownload = @()
$WhlCache = Get-WhlInventory -WhlDir $PipCacheWin -DistroName $DistroName


foreach ($pkg in $PyWheelsMissing) {
	$match = $WhlCache | Where-Object { "$($_['Name'])==$($_['Version'])" -eq $pkg.Name.ToLower() }
    if (-not $match) {
        Write-Host "⬇️ В temp/pip $($pkg.Name) не найден"
        #$PyWheelsToDownload += $pkg
    } else {
        Write-Host "✅ $($pkg.Name) уже ранее был скачен в temp/pip"
    }
}




# Генерация requirements_*.in по группам из $PyWheelsMissing
$PyTorchWheelsMissing = $PyWheelsMissing | Where-Object { $_.Source -eq "torch" } 
$PypiWheelsMissing  = $PyWheelsMissing | Where-Object { $_.Source -eq "pypi" } 

# Пути к *.in/.txt (Windows)
$ReqInTorchPathWin  = Join-Path $PipCacheWin "requirements_torch.in"
$ReqTxtTorchPathWin = Join-Path $PipCacheWin "requirements_torch.txt"
$ReqInPyPiPathWin   = Join-Path $PipCacheWin "requirements_pypi.in"
$ReqTxtPyPiPathWin  = Join-Path $PipCacheWin "requirements_pypi.txt"  

# Пути к *.in/.txt (WSL)
$ReqInTorchPathWsl = Convert-WindowsPathToWsl $ReqInTorchPathWin
$ReqTxtTorchPathWsl = Convert-WindowsPathToWsl $ReqTxtTorchPathWin
$ReqInPyPiPathWsl = Convert-WindowsPathToWsl $ReqInPyPiPathWin
$ReqTxtPyPiPathWsl = Convert-WindowsPathToWsl $ReqTxtPyPiPathWin



# Создаём *.in и компилируем *.txt через uv внутри WSL
if ($PyTorchWheelsMissing.Count -gt 0) {
	#Получаем список пакетов torch из $PyWheels и записываем в файл
	$PyWheelsTorch = ($PyWheels | Where-Object { $_.Source -eq "torch" }) | ForEach-Object { $_['Name'] }
    $PyWheelsTorch | Set-Content -Encoding UTF8 -Path $ReqInTorchPathWin
    wsl -d $DistroName -- bash -c "uv pip compile '$ReqInTorchPathWsl' --output-file '$ReqTxtTorchPathWsl' --extra-index-url https://download.pytorch.org/whl/cu118 > /dev/null 2>&1"
}

if ($PypiWheelsMissing.Count -gt 0) {
	#Получаем список пакетов pypi из $PyWheels и записываем в файл
	$PyWheelsPyPi = ($PyWheels | Where-Object { $_.Source -eq "pypi" }) | ForEach-Object { $_['Name'] }
    $PyWheelsPyPi | Set-Content -Encoding UTF8 -Path $ReqInPyPiPathWin
    wsl -d $DistroName -- bash -c "uv pip compile '$ReqInPyPiPathWsl' --output-file '$ReqTxtPyPiPathWsl' > /dev/null 2>&1"
}



#Следующий блок
###############################







########################


Write-Host "❌ СТОП ТЕСТ"; exit 1


<#
uv надо устанавливать до генерации req txt как whl пакет!

некоторые пакеты в метадата с большой буквы названия
#>
