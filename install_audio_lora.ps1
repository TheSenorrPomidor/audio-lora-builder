Write-Host "`nВерсия скрипта install_audio_lora.ps1 4.3"
# === install_audio_lora.ps1 ===

# === Настройка путей ===
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
<#
function Ensure-PyannoteAudioCli {
    Write-Host "🛠️ Проверка наличия CLI-команды pyannote-audio..."

    $TargetPath = "/usr/local/bin/pyannote-audio"
    $CheckCmd = "test -x '$TargetPath'"
    $null = wsl -d $DistroName -- bash -c "$CheckCmd"
    $Exists = $LASTEXITCODE -eq 0

    if ($Exists) {
        Write-Host "✅ CLI 'pyannote-audio' уже существует. Пропускаем."
        return
    }

    Write-Host "⚠️ CLI 'pyannote-audio' не найден. Добавляем вручную..."

    $ScriptContent = @"
#!/usr/bin/env python3
from pyannote.audio.cli import main

if __name__ == "__main__":
    main()
"@

    $TempWrapper = Join-Path $TempDir "pyannote-audio"
    $ScriptContent | Set-Content -Encoding UTF8 -Path $TempWrapper

    wsl -d $DistroName -- bash -c "sudo cp /mnt/d/VM/WSL2/audio-lora-builder/temp/pyannote-audio '$TargetPath' && sudo chmod +x '$TargetPath'"

    Write-Host "✅ CLI 'pyannote-audio' создан вручную и установлен в $TargetPath"
}
#>

# === 1. Проверка и удаление WSL-дистрибутива ===
Write-Host "`n1. 🔍 Проверяем наличие WSL-дистрибутива '$DistroName'..."
$existingDistros = wsl --list --quiet
if ($existingDistros -contains $DistroName) {
    $response = Read-Host "⚠️ Дистрибутив '$DistroName' уже существует. Удалить его и переустановить? [Y/N]"
    if ($response -eq "Y") {
        Write-Host "🧹 Удаляем существующий дистрибутив..."
		wsl --unregister $DistroName
    } else {
        Write-Host "⏭️ Прерываем установку."
        exit 0
    }
}
else {
	Write-Host "✅ Действующий $DistroName не найден"
}

# === 2. Получение rootfs ===
Write-Host "`n2. 📦 Поиск базового или финального rootfs"
if (Test-Path $FinalRootfs) {
    $ImportRootfs = $FinalRootfs
    Write-Host "✅ Найден финальный rootfs: $FinalRootfs"
} elseif (Test-Path $BaseRootfs) {
    $ImportRootfs = $BaseRootfs
    Write-Host "✅ Найден базовый rootfs: $BaseRootfs"
} else {
	
# === 2.1. Получение базового ubuntu2204_rootfs.tar.gz через распаковку .appx ===
    Write-Host "`n2.1. 📦️ Базовый и финальный rotfs не найден, качаем appxbundle ..."

	
    if (-not (Test-Path $BundleZipFile)) {
		$urlBundle = "https://aka.ms/wslubuntu2204"
        Write-Host "`n📦 Ищем/Скачиваем архив .appxbundle Ubuntu 22.04 (из $urlBundle в $BundleZipFile)"
        if (-not (Test-Path $TempDir)) { New-Item -ItemType Directory -Path $TempDir | Out-Null }
        Invoke-WebRequest -Uri $urlBundle -OutFile $BundleZipFile -UseBasicParsing
        Write-Host "✅ Закачка Appxbundle завершена ($BundleZipFile)"
    } else {
        Write-Host "📦 Appxbundle уже был ранее скачен ($BundleZipFile)"
    }

    # 2.2 Распаковка appxbundle
    Write-Host "`n📦 Распаковываем .appxbundle ($BundleZipFile) для извлечения Ubuntu_2204.1.7.0_x64.appx"
    try {
        Expand-Archive -Path $BundleZipFile -DestinationPath $BundleExtractPath -Force
        Write-Host "✅ Распаковка .appxbundle завершена."
    } catch {
        Write-Host "❌ Ошибка при распаковке .appxbundle: $_"
        exit 1
    }

    # 2.3 Распаковка Ubuntu_2204.1.7.0_x64.appx и извлечение install.tar.gz который является базовым rootfs
    Write-Host "`n📦 Распаковка Ubuntu_2204.1.7.0_x64.appx и извлечение install.tar.gz который является базовым rootfs..."
    try {
		$AppxFile = Join-Path $BundleExtractPath "Ubuntu_2204.1.7.0_x64.appx"
		$AppxZipFile = "$AppxFile.zip"
        Copy-Item -Path $AppxFile -Destination $AppxZipFile -Force
        Expand-Archive -Path $AppxZipFile -DestinationPath $BundleExtractPath -Force
        Remove-Item $AppxZipFile
		$BundleRootfs = Join-Path $BundleExtractPath "install.tar.gz" 
        if (-not (Test-Path $RootfsDir)) { New-Item -ItemType Directory -Path $RootfsDir | Out-Null }
        Move-Item -Path $BundleRootfs -Destination $BaseRootfs -Force
        Write-Host "✅ $BundleRootfs переименован и перемещен в: $BaseRootfs"
        $ImportRootfs = $BaseRootfs
    } catch {
        Write-Host "❌ Ошибка при распаковке .appx: $_"
        exit 1
    }
}

		
# === 3. Импорт WSL дистро ===

Write-Host "`n3. 💽 Импортируем WSL-дистрибутив из '$ImportRootfs'..."
wsl --import $DistroName "$ScriptDir\$DistroName" $ImportRootfs --version 2
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Ошибка при импорте дистрибутива."
    exit 1
}



# === 4. Восстановление DNS===
Write-Host "`n4. 🌐 Восстанавливаем DNS в WSL..."
wsl -d $DistroName -- bash -c "echo 'nameserver 8.8.8.8' > /etc/resolv.conf"
# и уменьшение таймаутов обращения к репозитариям и количество попыток 
#wsl -d $DistroName -- bash -c "echo 'Acquire::http::Timeout 1;' | sudo tee /etc/apt/apt.conf.d/99timeout"
#wsl -d $DistroName -- bash -c "echo 'Acquire::https::Timeout 1;' | sudo tee -a /etc/apt/apt.conf.d/99timeout"
#wsl -d $DistroName -- bash -c "echo 'Acquire::Retries 0;' | sudo tee -a /etc/apt/apt.conf.d/99timeout"


# === 5. Установка зависимостей ===
Write-Host "`n5. 📦 === Установка зависимостей ==="

# 5.1. Установка python3-pip, ffmpeg, dpkg-dev, unzip  (ffmpeg - для конвертации аудио, dpkg-dev для генерации Packages.gz, unzip для распаковки архивов pip пакетов)
Write-Host "5.1 📦 Установка python3-pip, ffmpeg, dpkg-dev, unzip"
$AptCacheWin = Join-Path $TempDir "apt"
$AptCacheWsl = Convert-WindowsPathToWsl $AptCacheWin
$Pkgs = @("python3-pip", "ffmpeg", "dpkg-dev", "unzip")

# Проверка, установлены ли пакеты из $Pkgs
$Missing = @()
foreach ($Pkg in $Pkgs) {
    $Status = wsl -d $DistroName -- bash -c "dpkg -s $Pkg 2>/dev/null | grep -E '^Status:'"
    if ($Status -match "ok installed") {
        Write-Host "✅ $Pkg уже установлен, пропускаем."
    } else {
        $Missing += $Pkg
    }
}
if ($Missing.Count -eq 0) {
    Write-Host "📦 Все пакеты уже установлены. Установка не требуется."
} else {
	# Проверка: есть ли .deb и Packages.gz
	$HasDebs = Test-Path "$AptCacheWin\*.deb"
	$HasIndex = Test-Path "$AptCacheWin\Packages.gz"

	if ($HasDebs -and $HasIndex) {
		Write-Host "📦 Найдены .deb и Packages.gz — используем оффлайн-репозиторий"
		
		#Отключаем официальные репо, что бы ставить только оффлайн
		wsl -d $DistroName -- bash -c "sudo mv /etc/apt/sources.list /etc/apt/sources.list.bak"

		# Добавляем temp/apt как локальный репозиторий
		$RepoLine = "deb [trusted=yes] file:/$($AptCacheWsl.TrimStart('/')) ./"
		$SourcesListPath = "/etc/apt/sources.list.d/local-temp-apt.list"
		$AddRepoCmd = "grep -Fxq '$RepoLine' $SourcesListPath 2>/dev/null || echo '$RepoLine' | sudo tee $SourcesListPath"

		wsl -d $DistroName -- bash -c "$AddRepoCmd"

		# apt update + установка
		wsl -d $DistroName -- bash -c "sudo apt-get update"
		wsl -d $DistroName -- bash -c "sudo apt-get install -y $($Pkgs -join ' ')"
		
		# Удаляем временный оффлайн-репозиторий после установки
		wsl -d $DistroName -- bash -c "sudo rm -f /etc/apt/sources.list.d/local-temp-apt.list"
		#Возврааем официальные репо, могут в дальнейшем пригодиться
		wsl -d $DistroName -- bash -c "sudo mv /etc/apt/sources.list.bak /etc/apt/sources.list"
	}
	else {
		Write-Host "📦 .deb или Packages.gz в temp не найдены — устанавливаем $($Pkgs -join ' ') онлайн и экспортируем"

		# Очистка кэша перед скачиванием
		wsl -d $DistroName -- bash -c "sudo rm -f /var/cache/apt/archives/*.deb"

		# Обновление и установка (гарантированная установка всех нужных пакетов)
		wsl -d $DistroName -- bash -c "sudo apt-get update"
		wsl -d $DistroName -- bash -c "sudo apt-get install -y $($Pkgs -join ' ')"
						

		# Явная загрузка .deb после установки (гарантированное сохранение установленных версий)
		wsl -d $DistroName -- bash -c "sudo apt-get install --download-only -y $($Pkgs -join ' ') --reinstall"				                   
									   
		# Создание каталога и перемещение .deb в temp/apt
		New-Item -ItemType Directory -Force -Path $AptCacheWin | Out-Null
		wsl -d $DistroName -- bash -c "mv /var/cache/apt/archives/*.deb '$AptCacheWsl/'"

		# Генерация Packages.gz
		wsl -d $DistroName -- bash -c "cd '$AptCacheWsl' && dpkg-scanpackages . /dev/null | gzip -c > Packages.gz"

		# Очистка кэша после скачивания
		wsl -d $DistroName -- bash -c "sudo rm -f /var/cache/apt/archives/*.deb"
		Write-Host "✅ Скачанные .deb и Packages.gz экспортированы в temp/apt"
	}
	Write-Host "📦 Проверка результата установки пакетов"
	foreach ($Pkg in $Pkgs) {
		$Status = wsl -d $DistroName -- bash -c "dpkg -s $Pkg 2>/dev/null | grep -E '^Status:'"
		if ($Status -match "ok installed") {
			Write-Host "✅ $Pkg установлен"
		} else {
			Write-Host "❌ $Pkg НЕ установлен"
		}
	}

}


# === 5.2 Установка CUDA Runtime 12.6 ===
Write-Host "`n5.2 📦 Установка CUDA Runtime 12.6 (через apt-get --download-only, оффлайн) v10"

# Пути
$CudaKeyUrl     = "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-archive-keyring.gpg"
$LocalCudaDebs = Join-Path $TempDir "cuda_debs"
$WslCudaDebs    = Convert-WindowsPathToWsl $LocalCudaDebs
$CudaKeyFile    = Join-Path $LocalCudaDebs "cuda-archive-keyring.gpg"
$WslCudaKeyFile = Convert-WindowsPathToWsl $CudaKeyFile
$Verify = ""
$Pkgs = @()



if ($WhisperImpl -eq 'faster-whisper') {
	#libcublas.so.12 ~500MB
	$Pkgs = @("libcublas-12-6", "libcublas-dev-12-6", "cuda-toolkit-12-config-common", "cuda-toolkit-12-6-config-common", "cuda-toolkit-config-common")
}
if ($WhisperImpl -eq 'whisperx') {
	#cuda-runtime-12-6 для WhisperX libcublas.so.12 ~500MB и cuda-runtime-12-6 +700MB = 1.2Gb
	$Pkgs = @("libcublas-12-6", "libcublas-dev-12-6", "cuda-toolkit-12-config-common", "cuda-toolkit-12-6-config-common", "cuda-toolkit-config-common", "cuda-runtime-12-6")
}	

# Проверка, установлены ли пакеты из $Pkgs
$Missing = @()
foreach ($Pkg in $Pkgs) {
    $Status = wsl -d $DistroName -- bash -c "dpkg -s $Pkg 2>/dev/null | grep -E '^Status:'"
    if ($Status -match "ok installed") {
        Write-Host "✅ $Pkg уже установлен, пропускаем."
    } else {
		Write-Host "⬇️ $Pkg требуется установить"
        $Missing += $Pkg
    }
}
if ($Missing.Count -eq 0) {
    Write-Host "✅ Весь CUDA Runtime уже установлен. Пропускаем установку CUDA."
}else {
	#Скачивание .deb, если нужно
	$HasCudaDebs = Test-Path "$LocalCudaDebs\*.deb"
	$HasIndex = Test-Path "$LocalCudaDebs\Packages.gz"
	
	if ($HasDebs -and $HasIndex) {
		Write-Host "📦 Найдены .deb и Packages.gz — используем оффлайн-репозиторий"
		
		#Отключаем официальные репо, что бы ставить только оффлайн
		wsl -d $DistroName -- bash -c "sudo mv /etc/apt/sources.list /etc/apt/sources.list.bak"

		# Добавляем temp/apt как локальный репозиторий
		$RepoLine = "deb [trusted=yes] file:/$($WslCudaDebs.TrimStart('/')) ./"
		$SourcesListPath = "/etc/apt/sources.list.d/local-temp-apt.list"
		$AddRepoCmd = "grep -Fxq '$RepoLine' $SourcesListPath 2>/dev/null || echo '$RepoLine' | sudo tee $SourcesListPath"

		wsl -d $DistroName -- bash -c "$AddRepoCmd"

		# apt update + установка
		wsl -d $DistroName -- bash -c "sudo apt-get update"
		wsl -d $DistroName -- bash -c "sudo apt-get install -y $($Pkgs -join ' ')"
		
		# Удаляем временный оффлайн-репозиторий после установки
		wsl -d $DistroName -- bash -c "sudo rm -f /etc/apt/sources.list.d/local-temp-apt.list"
		#Возврааем официальные репо, могут в дальнейшем пригодиться
		wsl -d $DistroName -- bash -c "sudo mv /etc/apt/sources.list.bak /etc/apt/sources.list"
	}
	else {
		Write-Host "📦 .deb или Packages.gz в temp не найдены — устанавливаем CUDA онлайн и экспортируем"

		if (-not (Test-Path $LocalCudaDebs)) { New-Item -ItemType Directory -Path $LocalCudaDebs | Out-Null }
		#Скачиваем GPG-ключ (если не скачан)
		if (!(Test-Path $CudaKeyFile)) {
			Write-Host "⬇️ Скачиваем GPG-ключ NVIDIA..."
			Invoke-WebRequest -Uri $CudaKeyUrl -OutFile $CudaKeyFile
		} else {
			Write-Host "📦 GPG-ключ уже скачан, пропускаем."
		}
		
		#Добавляем репозиторий CUDA (signed-by)
		wsl -d $DistroName -- bash -c "echo 'deb [signed-by=$WslCudaKeyFile] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /' | sudo tee /etc/apt/sources.list.d/cuda.list"

		# Очистка кэша перед скачиванием
		wsl -d $DistroName -- bash -c "sudo rm -f /var/cache/apt/archives/*.deb"

		# Обновление и установка (гарантированная установка всех нужных пакетов)
		wsl -d $DistroName -- bash -c "sudo apt-get update"
		wsl -d $DistroName -- bash -c "sudo apt-get install -y $($Pkgs -join ' ')"
						
		# Явная загрузка .deb после установки (гарантированное сохранение установленных версий)
		wsl -d $DistroName -- bash -c "sudo apt-get install --download-only -y $($Pkgs -join ' ') dpkg-dev --reinstall"				                   
									   
		# Создание каталога и перемещение .deb в temp/apt
		New-Item -ItemType Directory -Force -Path $AptCacheWin | Out-Null
		wsl -d $DistroName -- bash -c "mv /var/cache/apt/archives/*.deb '$WslCudaDebs/'"

		# Генерация Packages.gz
		wsl -d $DistroName -- bash -c "cd '$WslCudaDebs' && dpkg-scanpackages . /dev/null | gzip -c > Packages.gz"

		# Удаление ссылки на репозиторий CUDA в WSL
		Write-Host "🧼 Удаление ссылки на репозиторий CUDA в WSL..."
		wsl -d $DistroName -- bash -c "rm -f /etc/apt/sources.list.d/cuda*.list"
		
		Write-Host "✅ Скачанные .deb и Packages.gz экспортированы в temp/cuda_debs"
	}
	Write-Host "📦 Проверка результата установки пакетов CUDA"
	foreach ($Pkg in $Pkgs) {
		$Status = wsl -d $DistroName -- bash -c "dpkg -s $Pkg 2>/dev/null | grep -E '^Status:'"
		if ($Status -match "ok installed") {
			Write-Host "✅ $Pkg установлен"
		} else {
			Write-Host "❌ $Pkg НЕ установлен"
		}
	}
}



# === 5.3 Установка cuDNN ===
Write-Host "5.3 📦 Установка cuDNN"

# Пути
$LocalCuDnnDebs = Join-Path $TempDir "cudnn_debs"
$CuDnnDebUrl = "https://developer.download.nvidia.com/compute/cudnn/9.10.2/local_installers/cudnn-local-repo-ubuntu2204-9.10.2_1.0-1_amd64.deb"
$CuDnnDebFile = Join-Path $LocalCuDnnDebs "cudnn-local-repo-ubuntu2204-9.10.2_1.0-1_amd64.deb"
$CuDnnWslPath = "/root/cudnn"
$CuDnnWslDeb = "$CuDnnWslPath/cudnn.deb"

# Проверка: установлен ли cuDNN
wsl -d $DistroName -- bash -c "dpkg -l | grep -q libcudnn"
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ cuDNN уже установлен в WSL — пропускаем установку."
}
else {
    #Скачивание .deb, если нужно
	#Отключаем официальные репо, что бы не мешалось
	wsl -d $DistroName -- bash -c "sudo mv /etc/apt/sources.list /etc/apt/sources.list.bak"
	$HasCudnnDebs = Test-Path "$LocalCudaDebs\*.deb"
	if (-Not $HasCudnnDebs) {
        Write-Host "⬇️ в $LocalCuDnnDebs нет файлов .deb, Скачиваем deb контейнер $CuDnnDebUrl..."
		if (-not (Test-Path $LocalCuDnnDebs)) { New-Item -ItemType Directory -Path $LocalCuDnnDebs | Out-Null }
        Invoke-WebRequest -Uri $CuDnnDebUrl -OutFile $CuDnnDebFile
    } else {
        Write-Host "📦 .deb-файл уже скачан, пропускаем загрузку."
    }

    #Установка cuDNN из .deb-репозитория внутри WSL
    Write-Host "📦 Устанавливаем cuDNN локально в WSL..."

    #Создаём временную папку в WSL
    wsl -d $DistroName -- bash -c "mkdir -p $CuDnnWslPath"

    #Удаляем старый .deb в WSL (если есть)
    wsl -d $DistroName -- bash -c "rm -f $CuDnnWslDeb"

    #Копируем .deb из Windows в WSL
	#Формируем путь из Windows-пути в WSL-путь через /mnt/
	$CuDnnDebFileWsl = Convert-WindowsPathToWsl $CuDnnDebFile

	#Копируем .deb из Windows в WSL
	wsl -d $DistroName -- bash -c "cp '$CuDnnDebFileWsl' '$CuDnnWslDeb'"


    #Устанавливаем локальный репозиторий (создаёт /var/cudnn-local-repo-...)
    wsl -d $DistroName -- bash -c "dpkg -i $CuDnnWslDeb"

    #Копируем GPG-ключ, чтобы apt доверял локальному репо
    wsl -d $DistroName -- bash -c "cp /var/cudnn-local-repo-ubuntu2204-9.10.2/cudnn-*-keyring.gpg /usr/share/keyrings/"

	#Обновляем apt и устанавливаем cuDNN-пакеты (указаны конкретные реализации)
	wsl -d $DistroName -- bash -c "apt-get update"
	wsl -d $DistroName -- bash -c "DEBIAN_FRONTEND=noninteractive apt-get -y install libcudnn9-cuda-12 libcudnn9-dev-cuda-12"


    #Очистка временной папки в WSL
    Write-Host "🧹 Удаляем временные файлы из WSL..."
    wsl -d $DistroName -- bash -c "rm -f $CuDnnWslDeb"
    wsl -d $DistroName -- bash -c "rm -rf $CuDnnWslPath"

	#Очистка всех возможных следов локального репозитория cuDNN
	Write-Host "🧼 Удаляем все возможные хвосты локального APT-репозитория cuDNN..."
	$cleanCudnn = @'
	rm -f /etc/apt/sources.list.d/cudnn*.list
	rm -f /usr/share/keyrings/cudnn-local-archive-keyring.gpg
	rm -rf /var/cudnn*
'@
	wsl -d $DistroName -- bash -c "$cleanCudnn"
	Write-Host "🔄 Обновляем APT после удаления локального репозитория cuDNN..."
	wsl -d $DistroName -- bash -c "apt-get update"
	
	#Возврааем официальные репо, могут в дальнейшем пригодиться
	wsl -d $DistroName -- bash -c "sudo mv /etc/apt/sources.list.bak /etc/apt/sources.list"	
		
    #Финальная проверка
    wsl -d $DistroName -- bash -c "dpkg -l | grep -q libcudnn"
    if ($LASTEXITCODE -eq 0) {

        Write-Host "✅ cuDNN успешно установлен в WSL и репозиторий удалён."
    } else {
        Write-Host "❌ Установка cuDNN завершилась с ошибкой."
    }
}




	
Write-Host "`n5.4 📦 Установка Python-библиотек (torch, $WhisperImpl) из temp\pip (Windows)"
$PipCacheWin = Join-Path $TempDir "pip"
$PipCacheWsl = Convert-WindowsPathToWsl $PipCacheWin

$PyWheels = @(
  @{ Name = "whisperx==3.3.1";       Source = "torch"; Impl = "whisperx" },
  @{ Name = "transformers==4.28.1";  Source = "torch"; Impl = "whisperx" },
  @{ Name = "librosa==0.10.0";       Source = "torch"; Impl = "whisperx" },
  @{ Name = "pyannote-audio==3.3.2"; Source = "torch"; Impl = "whisperx" },
  @{ Name = "hydra-core==1.3.2";     Source = "torch"; Impl = "whisperx" },
  @{ Name = "faster-whisper";        Source = "pypi";  Impl = "faster-whisper" }

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

if ($PyWheelsMissing.Count -gt 0) {

	#Удаляем кэш который мог быть в WSL
	wsl -d $DistroName -- bash -c "rm -rf ~/.cache/pip"
	# Проверяем наличие ранее скаченных пакетов .whl в temp/pip
	$PyWheelsToDownload = @()
	$WhlCache = Get-WhlInventory -WhlDir $PipCacheWin -DistroName $DistroName


	foreach ($pkg in $PyWheelsMissing) {
		$match = $WhlCache | Where-Object { "$($_['Name'])==$($_['Version'])" -eq $pkg.Name.ToLower() }
		if (-not $match) {
			Write-Host "⬇️ В temp/pip $($pkg.Name) не найден"
			$PyWheelsToDownload += $pkg
		} else {
			Write-Host "✅ $($pkg.Name) уже ранее был скачен в temp/pip"
		}
	}





	#Устанавливаем uv
	$UvWheel = Get-ChildItem $PipCacheWin -Filter "uv-*.whl" | Select-Object -First 1
	if (-not $UvWheel) {
		wsl -d $DistroName -- bash -c "pip download uv -d '$PipCacheWsl'"
		$UvWheel = Get-ChildItem $PipCacheWin -Filter "uv-*.whl" | Select-Object -First 1
	}
	wsl -d $DistroName -- bash -c "pip install '$($PipCacheWsl)/$($UvWheel.Name)' --no-index --find-links='$PipCacheWsl' > /dev/null 2>&1"




	
	@("torch", "pypi") | ForEach-Object {
    $group = $_

    $inPathWin  = Join-Path $PipCacheWin  "requirements_${group}.in"
    $txtPathWin = Join-Path $PipCacheWin  "requirements_${group}.txt"
    $inPathWsl  = Convert-WindowsPathToWsl $inPathWin
    $txtPathWsl = Convert-WindowsPathToWsl $txtPathWin

    $packages = $PyWheels | Where-Object {
        $_.Source -eq $group -and $_.Impl -eq $WhisperImpl
    } | ForEach-Object { $_['Name'] }

    if ($packages.Count -eq 0) { return }

    $packages | Set-Content -Encoding UTF8 -Path $inPathWin



	
	$toDownload = $PyWheelsToDownload | Where-Object { $_.Source -eq $group }
	if ((Test-Path $txtPathWin) -and ($toDownload.Count -eq 0)) {
		Write-Host "`n📄 Все ключевые пакеты WHL по источнику $group скачены и файл requirements_${group}.txt уже есть, Генерация нового requirements_${group}.txt не требуется..."
		} else {
				$compileCmd = "uv pip compile '$inPathWsl' --output-file '$txtPathWsl'"
				if ($group -eq "torch") {
					$compileCmd += " --extra-index-url https://download.pytorch.org/whl/cu118"
				}		
				Write-Host "`n📄 Генерация requirements_${group}.txt..."
				wsl -d $DistroName -- bash -c "$compileCmd > /dev/null 2>&1"
				Write-Host "🌐 Загрузка зависимостей $group..."
				$downloadCmd = "pip download -r '$txtPathWsl' -d '$PipCacheWsl'"
				if ($group -eq "torch") {
					$downloadCmd += " --extra-index-url https://download.pytorch.org/whl/cu118"
				}
				wsl -d $DistroName -- bash -c "$downloadCmd"
			}
		Write-Host "📦 Установка $group-пакетов..."
		wsl -d $DistroName -- bash -c "pip install --no-index --find-links='$PipCacheWsl' -r '$txtPathWsl'"
	}




	# Компилируем .tar.gz и .zip → .whl
	$Archives = Get-ChildItem -Path $PipCacheWin -Include *.tar.gz,*.zip -Recurse
	foreach ($pkg in $Archives) {
		$pkgPathWsl = Convert-WindowsPathToWsl $pkg.FullName
		Write-Host "🛠️ Компиляция: $($pkg.Name)"
		wsl -d $DistroName -- bash -c "pip wheel '$pkgPathWsl' --no-deps --wheel-dir '$PipCacheWsl' > /dev/null 2>&1"
		Remove-Item $pkg.FullName -Force
	}

	#Удаляем кэш который накопился в WSL
	wsl -d $DistroName -- bash -c "rm -rf ~/.cache/pip"

	#Проверить установились ли пакеты .whl в WSL после фазы установки пакетов
	foreach ($pkg in $PyWheels) {
		$IsForThisImpl = ($pkg.Impl -eq "all" -or $pkg.Impl -eq $WhisperImpl)
		if ($IsForThisImpl) {

			$DepName = $pkg.Name -split '==|\+' | Select-Object -First 1
			$DepVersion = $pkg.Name -split '==|\+' | Select-Object -Skip 1 | Select-Object -First 1
			$IsInstalled = wsl -d $DistroName -- bash -c "pip show $DepName > /dev/null && echo ok"
			if ($IsInstalled -eq "ok") {
				Write-Host "✅ $($pkg.Name) удалось установить — всё хорошо"
			} else {
				Write-Host "❌ $($pkg.Name) установить не удалось"
			}
		}
	}
	

}
else {
	Write-Host "✅ Все необходимые Python-библиотеки установлены"
}


# === 5.5 Предзагрузка и кэширование модели Whisper large-v3 (faster-whisper) для CPU и GPU ===
#Определяем переменные для блока
Write-Host "`n5.5 🧠 Предзагрузка и кэширование модели $WhisperImpl для CPU и GPU"
Write-Host "📦 Определение переменных"
$ModelCacheWin = Join-Path $TempDir "huggingface\$WhisperImpl"
$ModelCacheWinWsl = Convert-WindowsPathToWsl $ModelCacheWin
$ModelCacheLocalWsl = "/root/.cache/huggingface/hub"
$ModelFullName = ""
$WhisperCacheSearchPattern = ""
$PythonLoadScript = ""

if ($WhisperImpl -eq "faster-whisper") {
	$ModelFullName = "Faster-Whisper large-v3"
	$WhisperCacheSearchPattern = "*whisper*large*v3*"
	$PythonLoadScript = @"
import os
os.environ['HF_HOME'] = r'$ModelCacheWinWsl'
from faster_whisper import WhisperModel

print('(python) 🔄 Кэшируем Faster Whisper large-v3 на CPU...')
WhisperModel('large-v3', device='cpu', compute_type='int8')
print('(python) ✅ Модель закэширована для CPU.')

try:
	print('(python) 🔄 Кэшируем Faster Whisper large-v3 на GPU (если поддерживается)...')
	WhisperModel('large-v3', device='cuda', compute_type='int8_float16')
	print('(python) ✅ Модель закэширована для GPU.')
except Exception as e:
	print('(python) ⚠️ Не удалось закэшировать для GPU: ' + str(e))
"@
}
if ($WhisperImpl -eq "whisperx") {
	$ModelFullName = "WhisperX large-v3"
	$WhisperCacheSearchPattern = "*whisper*large*v3*"	
	$PythonLoadScript = @"
import os
os.environ['HF_HOME'] = r'$ModelCacheWinWsl'

import whisperx

print('(python) 🔄 Кэшируем WhisperX large-v3 на CPU...')
model = whisperx.load_model("large-v3", device="cpu", compute_type='int8')
print('(python) ✅ Модель WhisperX закэширована (CPU).')

try:
    print('(python) 🔄 Кэшируем WhisperX large-v3 на GPU (если поддерживается)...')
	model = whisperx.load_model("large-v3", device="cuda", compute_type='int8_float16')
    print('(python) ✅ Модель WhisperX закэширована (GPU).')
except Exception as e:
	print('(python) ⚠️ Не удалось закэшировать WhisperX для GPU: ' + str(e))
"@
}


#Проверка установленного кэша модели в WSL)
Write-Host "📦 Проверка установленного кэша модели $ModelFullName в WSL..."
$CheckModelCmd = "find $ModelCacheLocalWsl -type d -iname $WhisperCacheSearchPattern | grep -q ."
wsl -d $DistroName -- bash -c "$CheckModelCmd"
$ModelCached = $LASTEXITCODE

if ($ModelCached -eq 0) {
	Write-Host "✅ Кэш $ModelFullName уже установлен на WSL. Пропускаем загрузку."
}
else {
	#Проверяем кэш в temp на Windows
	Write-Host "📦 Кэш в WSL не найден, ищем кэш в temp: $ModelCacheWin"
	$CheckModelCmd = "find $ModelCacheWinWsl -type d -iname $WhisperCacheSearchPattern | grep -q ."
	wsl -d $DistroName -- bash -c "$CheckModelCmd"
	$ModelDownloaded = $LASTEXITCODE
	
	if ($ModelDownloaded -eq 0) {
		Write-Host "📦 Кэш модели $ModelFullName найден. Копируем в WSL..."
		wsl -d $DistroName -- bash -c "mkdir -p '$ModelCacheLocalWsl' && cp -r '$ModelCacheWinWsl/hub/'* '$ModelCacheLocalWsl/'"

		Write-Host "✅ Кэш успешно скопирован Windows => WSL."
	}
	else {
		Write-Host "📦 Кэш модели $ModelFullName не найден. Скачиваем модель с huggingface"
		$TempPreloadPyFile = Join-Path $TempDir "preload_${WhisperImpl}.py"
		$TempPreloadPyFileWsl = Convert-WindowsPathToWsl $TempPreloadPyFile	
		$PythonLoadScript | Out-File -FilePath $TempPreloadPyFile -Encoding UTF8

		wsl -d $DistroName -- bash -c "python3 '$TempPreloadPyFileWsl'"
		Remove-Item $TempPreloadPyFile -Force

		Write-Host "📦 Кэш модели $ModelFullName скачен. Копируем в WSL..."
		wsl -d $DistroName -- bash -c "mkdir -p '$ModelCacheLocalWsl' && cp -r '$ModelCacheWinWsl/hub/'* '$ModelCacheLocalWsl/'"
		Write-Host "✅ Кэш скачен и скопирован Windows => WSL."
	}
}






# === 5.6 ⚙️ Генерация конфигурации путей ===
Write-Host "`n5.6 ⚙️ Генерируем конфигурацию путей..."

$AudioSrcDir = Join-Path $ScriptDir "audio_src"

# Если папки нет — создадим
if (-not (Test-Path $AudioSrcDir)) {
    Write-Host "📁 Папка audio_src не найдена. Создаём..."
    New-Item -ItemType Directory -Path $AudioSrcDir | Out-Null
}

# Преобразуем Windows-путь в WSL-совместимый
$WSL_WIN_AUDIO_SRC = Convert-WindowsPathToWsl $AudioSrcDir

# Готовим содержимое .env-файла
$escapedContent = "WIN_AUDIO_SRC=$WSL_WIN_AUDIO_SRC"
$WSL_ENV_FILE = "/root/audio-lora-builder/config/env.vars"

# Пишем в WSL напрямую
$escapedCommand = "mkdir -p /root/audio-lora-builder/config && echo '$escapedContent' > $WSL_ENV_FILE"
wsl -d $DistroName -- bash -c "$escapedCommand"

Write-Host "✅ Конфигурация сохранена в WSL: $WSL_ENV_FILE"


# === 6. Создание снапшота ===
Write-Host "`n8. 💽 Создаём rootfs-снапшот дистрибутива..."
wsl --export $DistroName $FinalRootfs
Write-Host "✅ Снапшот сохранён: $FinalRootfs"

# === 7. Инструкция ===
Write-Host "`n✅ Установка завершена!"
Write-Host "📥 Для запуска распознавания: wsl -d $DistroName -- python3 ~/audio-lora-builder/process_audio.py"






#Write-Host "❌ СТОП ТЕСТ"; exit 1


