Write-Host "`nВерсия скрипта 3.8"
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

# === 2. Получение rootfs ===
Write-Host "`n2. 📦 Поиск базового или финального rootfs"
if (Test-Path $FinalRootfs) {
    $ImportRootfs = $FinalRootfs
    Write-Host "`n✅ Найден финальный rootfs: $FinalRootfs"
} elseif (Test-Path $BaseRootfs) {
    $ImportRootfs = $BaseRootfs
    Write-Host "`n✅ Найден базовый rootfs: $BaseRootfs"
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
Write-Host "`n3. 💽 Импортируем WSL-дистрибутив из $BaseRootfs ..."
wsl --import $DistroName "$ScriptDir\$DistroName" $ImportRootfs --version 2
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Ошибка при импорте дистрибутива."
    exit 1
}


# === 4. Восстановление DNS и уменьшение таймаутов обращения к репозитариям и количество попыток ===
Write-Host "`n4. 🌐 Восстанавливаем DNS в WSL..."
wsl -d $DistroName -- bash -c "echo 'nameserver 8.8.8.8' > /etc/resolv.conf"

#wsl -d $DistroName -- bash -c "echo 'Acquire::http::Timeout 1;' | sudo tee /etc/apt/apt.conf.d/99timeout"
#wsl -d $DistroName -- bash -c "echo 'Acquire::https::Timeout 1;' | sudo tee -a /etc/apt/apt.conf.d/99timeout"
#wsl -d $DistroName -- bash -c "echo 'Acquire::Retries 0;' | sudo tee -a /etc/apt/apt.conf.d/99timeout"


# === 5. Установка зависимостей ===
Write-Host "5. 📦 === Установка зависимостей ==="

# === 5.1 Установка CUDA Runtime 12.6 (через apt-get --download-only, оффлайн) v10 ===
Write-Host "`n5.1 📦 Установка CUDA Runtime 12.6 (через apt-get --download-only, оффлайн) v10"

# Пути
$CudaKeyUrl     = "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-archive-keyring.gpg"
$LocalCudaDebs = Join-Path $TempDir "cuda_debs"
$WslCudaDebs    = Convert-WindowsPathToWsl $LocalCudaDebs
$CudaKeyFile    = Join-Path $LocalCudaDebs "cuda-archive-keyring.gpg"
$WslCudaKeyFile = Convert-WindowsPathToWsl $CudaKeyFile

# 5.1.1 Проверка: установлен ли уже libcublas.so.12
$CudaTest = wsl -d $DistroName -- bash -c "ldconfig -p | grep libcublas.so.12"
if ($CudaTest) {
    Write-Host "✅ CUDA Runtime уже установлен. Пропускаем."
}
else {
	# 5.1.2. Скачивание .deb, если нужно
	$HasCudaDebs = Test-Path "$LocalCudaDebs\*.deb"
	if (-Not $HasCudaDebs) {
		Write-Host "⬇️ в $LocalCudaDebs нет файлов .deb, Скачиваем локальный .deb-репозиторий CUDA..."
		if (-not (Test-Path $LocalCudaDebs)) { New-Item -ItemType Directory -Path $LocalCudaDebs | Out-Null }
		# 5.1.2.1. Скачиваем GPG-ключ (если не скачан)
		if (!(Test-Path $CudaKeyFile)) {
			Write-Host "⬇️ Скачиваем GPG-ключ NVIDIA..."
			Invoke-WebRequest -Uri $CudaKeyUrl -OutFile $CudaKeyFile
		} else {
			Write-Host "📦 GPG-ключ уже скачан, пропускаем."
		}	

		# 5.1.2.2. Добавляем репозиторий CUDA (signed-by)
		wsl -d $DistroName -- bash -c "echo 'deb [signed-by=$WslCudaKeyFile] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /' | sudo tee /etc/apt/sources.list.d/cuda.list"

		# 5.1.2.3. Обновляем apt
		wsl -d $DistroName -- bash -c "apt-get update"

		# 5.1.2.4. Скачиваем .deb-файлы CUDA и зависимостей (в offline-кеш)
		Write-Host "⬇️ Скачиваем .deb-файлы CUDA Runtime 12.6 через apt download в WSL..."
		wsl -d $DistroName -- bash -c "apt-get -y --reinstall install --download-only -o=dir::cache=$WslCudaDebs libcublas-12-6 libcublas-dev-12-6 cuda-toolkit-12-config-common cuda-toolkit-12-6-config-common cuda-toolkit-config-common"

		# 5.1.2.5. удаляем мусор из папки с кэшэм и оставляем там только .deb файлы 
		#Что она делает:
		#cd — переходит в целевую директорию.
		#find archives -name '*.deb' -exec mv -t . {} + — находит .deb внутри archives/ и переносит в текущую директорию.
		#rm -rf archives — удаляет папку archives.
		#rm -f *.bin — удаляет .bin-файлы в текущей директории.
		#✅ Всё в одной строке.
		wsl -d $DistroName -- bash -c "cd $WslCudaDebs && find archives -name '*.deb' -exec mv -t . {} + && rm -rf archives && rm -f *.bin"
		
		
		Write-Host "✅ .deb-файлы CUDA сохранены: $LocalCudaDebs"
    } else {
        Write-Host "📦 .deb-файл уже скачан, пропускаем загрузку."
    }

	# 5.1.3. Устанавливаем .deb в WSL
	wsl -d $DistroName -- bash -c "dpkg -i $WslCudaDebs/*.deb"

	# 5.1.4. Проверка
	$Verify = wsl -d $DistroName -- bash -c "ldconfig -p | grep libcublas.so.12"
	if ($Verify) {
		Write-Host "✅ CUDA Runtime 12.6 успешно установлен и доступен."
	} else {
		Write-Host "❌ CUDA Runtime установлен, но libcublas.so.12 не найден."
	}
	# 5.1.5. Удаление ссылки на репозиторий CUDA в WSL
	Write-Host "🧼 Удаление ссылки на репозиторий CUDA в WSL..."
	$cleanCudnn = @'
	rm -f /etc/apt/sources.list.d/cuda*.list
'@
}

# === 5.2 Установка cuDNN v6 (официальный .deb + apt, с полной очисткой) ===
Write-Host "5.2 📦 Установка cuDNN v6 (через .deb + apt, с удалением репо)"

# Пути
$LocalCuDnnDebs = Join-Path $TempDir "cudnn_debs"
$CuDnnDebUrl = "https://developer.download.nvidia.com/compute/cudnn/9.10.2/local_installers/cudnn-local-repo-ubuntu2204-9.10.2_1.0-1_amd64.deb"
$CuDnnDebFile = Join-Path $LocalCuDnnDebs "cudnn-local-repo-ubuntu2204-9.10.2_1.0-1_amd64.deb"
$CuDnnWslPath = "/root/cudnn"
$CuDnnWslDeb = "$CuDnnWslPath/cudnn.deb"

# 5.2.1. Проверка: установлен ли cuDNN
wsl -d $DistroName -- bash -c "dpkg -l | grep -q libcudnn"
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ cuDNN уже установлен в WSL — пропускаем установку."
}
else {
    # 5.2.2. Скачивание .deb, если нужно
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

    # 5.2.4. Установка cuDNN из .deb-репозитория внутри WSL
    Write-Host "📦 Устанавливаем cuDNN локально в WSL..."

    # 5.2.4.1 Создаём временную папку в WSL
    wsl -d $DistroName -- bash -c "mkdir -p $CuDnnWslPath"

    # 5.2.4.2 Удаляем старый .deb в WSL (если есть)
    wsl -d $DistroName -- bash -c "rm -f $CuDnnWslDeb"

    # 5.2.4.3 Копируем .deb из Windows в WSL
	# Формируем путь из Windows-пути в WSL-путь через /mnt/
	$CuDnnDebFileWsl = Convert-WindowsPathToWsl $CuDnnDebFile

	# Копируем .deb из Windows в WSL
	wsl -d $DistroName -- bash -c "cp '$CuDnnDebFileWsl' '$CuDnnWslDeb'"


    # 5.2.4.4 Устанавливаем локальный репозиторий (создаёт /var/cudnn-local-repo-...)
    wsl -d $DistroName -- bash -c "dpkg -i $CuDnnWslDeb"

    # 5.2.4.5 Копируем GPG-ключ, чтобы apt доверял локальному репо
    wsl -d $DistroName -- bash -c "cp /var/cudnn-local-repo-ubuntu2204-9.10.2/cudnn-*-keyring.gpg /usr/share/keyrings/"

	# 5.2.4.6 Обновляем apt и устанавливаем cuDNN-пакеты (указаны конкретные реализации)
	wsl -d $DistroName -- bash -c "apt-get update"
	wsl -d $DistroName -- bash -c "DEBIAN_FRONTEND=noninteractive apt-get -y install libcudnn9-cuda-12 libcudnn9-dev-cuda-12"


    # 5.2.5. Очистка временной папки в WSL
    Write-Host "🧹 Удаляем временные файлы из WSL..."
    wsl -d $DistroName -- bash -c "rm -f $CuDnnWslDeb"
    wsl -d $DistroName -- bash -c "rm -rf $CuDnnWslPath"

	# 5.2.6. Очистка всех возможных следов локального репозитория cuDNN
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
		
    # 5.2.7. Финальная проверка
    wsl -d $DistroName -- bash -c "dpkg -l | grep -q libcudnn"
    if ($LASTEXITCODE -eq 0) {

        Write-Host "✅ cuDNN успешно установлен в WSL и репозиторий удалён."
    } else {
        Write-Host "❌ Установка cuDNN завершилась с ошибкой."
    }
}



# 5.3. Установка python3-pip и ffmpeg (оффлайн с зависимостями через локальный репозиторий)
Write-Host "`n5.3 📦 Установка python3-pip и ffmpeg"
$AptCacheWin = Join-Path $TempDir "apt"
$AptCacheWsl = Convert-WindowsPathToWsl $AptCacheWin
$Pkgs = @("python3-pip", "ffmpeg")

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
    return
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
		Write-Host "📦 .deb или Packages.gz не найдены — устанавливаем онлайн и экспортируем"

		# Очистка кэша перед скачиванием
		wsl -d $DistroName -- bash -c "sudo rm -f /var/cache/apt/archives/*.deb"

		# Обновление и установка (гарантированная установка всех нужных пакетов)
		wsl -d $DistroName -- bash -c "sudo apt-get update"
		wsl -d $DistroName -- bash -c "sudo apt-get install -y $($Pkgs -join ' ') dpkg-dev"
						

		# Явная загрузка .deb после установки (гарантированное сохранение установленных версий)
		wsl -d $DistroName -- bash -c "sudo apt-get install --download-only -y $($Pkgs -join ' ') dpkg-dev --reinstall"				                   
									   
		# Создание каталога и перемещение .deb в temp/apt
		New-Item -ItemType Directory -Force -Path $AptCacheWin | Out-Null
		wsl -d $DistroName -- bash -c "mv /var/cache/apt/archives/*.deb '$AptCacheWsl/'"

		# Генерация Packages.gz
		wsl -d $DistroName -- bash -c "cd '$AptCacheWsl' && dpkg-scanpackages . /dev/null | gzip -c > Packages.gz"

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

	



# === 5.4 Установка Python-библиотек (torch, faster-whisper) из temp\pip (Windows) ===
Write-Host "`n5.4 📦 Установка Python-библиотек (torch, faster-whisper) из temp\pip (Windows)"

$PipCacheWin = Join-Path $TempDir "pip"
$PipCacheWsl = Convert-WindowsPathToWsl $PipCacheWin

$TorchWheel = "torch*.whl"
$FWheel = "faster_whisper*.whl"

# 5.4.1 Проверяем, установлены ли Python-библиотеки
$PipCheck = "pip show torch > /dev/null 2>&1 && pip show faster-whisper > /dev/null 2>&1"
wsl -d $DistroName -- bash -c "$PipCheck"
$PyDepsOk = $LASTEXITCODE

if ($PyDepsOk -eq 0) {
    Write-Host "5.4.1 ✅ Python-библиотеки torch и faster-whisper уже установлены. Пропускаем установку и скачивание wheel-файлов."
} else {
    # Только если пакеты не установлены, проверяем wheel-файлы и при необходимости скачиваем!
    $TorchExists = Test-Path (Join-Path $PipCacheWin $TorchWheel)
    $FWExists = Test-Path (Join-Path $PipCacheWin $FWheel)

    if (-not $TorchExists -or -not $FWExists) {
        Write-Host "📦 Wheel-файлы отсутствуют, скачиваем их (интернет только первый раз)..."
        wsl -d $DistroName -- bash -c "pip download torch --index-url https://download.pytorch.org/whl/cu118 -d '$PipCacheWsl'"
        wsl -d $DistroName -- bash -c "pip download faster-whisper -d '$PipCacheWsl'"
    } else {
        Write-Host "📦 Все wheel-файлы уже есть в temp\pip, скачивание не требуется."
    }

    Write-Host "📦 Устанавливаем Python-библиотеки из temp\pip (.whl)..."
    wsl -d $DistroName -- bash -c "pip install --no-index --find-links='$PipCacheWsl' torch faster-whisper"
}





# === 5.5 Предзагрузка и кэширование модели Whisper large-v3 (faster-whisper) для CPU и GPU ===
Write-Host "`n5.5 📦 Предзагрузка и кэширование модели Whisper large-v3 (faster-whisper) для CPU и GPU"
$ModelCacheWin = Join-Path $TempDir "huggingface\whisper"
$ModelCacheWinWsl = Convert-WindowsPathToWsl $ModelCacheWin
$ModelCacheLocalWsl = "/root/.cache/huggingface/hub"


# 5.5.1 Проверка установленного кэша модели в WSL (ищем любые *whisper*large*v3* папки)
Write-Host "5.5.1  📦 Проверка установленного кэша модели в WSL (ищем любые *whisper*large*v3* папки)"
$CheckModelCmd = "find $ModelCacheLocalWsl -type d -iname '*whisper*large*v3*' | grep -q ."
wsl -d $DistroName -- bash -c "$CheckModelCmd"
$ModelCached = $LASTEXITCODE

if ($ModelCached -eq 0) {
    Write-Host "✅ Кэш Whisper large-v3 уже установлен на WSL. Пропускаем загрузку."
}
else {
    # 5.5.2 Проверяем кэш в temp на Windows
	Write-Host "5.5.2  📦 Кэш в WSL не найден, ищем кэш в $ModelCacheWin"
    $ModelExistsInTemp = Get-ChildItem -Path $ModelCacheWin -Directory -Recurse -ErrorAction SilentlyContinue | Where-Object { $_.Name -like "*whisper*large*v3*" }
    if ($ModelExistsInTemp) {
        Write-Host "📦 Кэш модели Whisper large-v3 найден. Копируем в WSL..."
		#Отладка
		Write-Host "`n🔎 Отладка: путь скрипта preload внутри WSL: TempPreloadWsl = $TempPreloadWsl, TempDir = $TempDir "
		#
		wsl -d $DistroName -- bash -c "mkdir -p '$ModelCacheLocalWsl' && cp -r '$ModelCacheWinWsl/hub/'* '$ModelCacheLocalWsl/'"

        Write-Host "✅ Кэш успешно скопирован Windows => WSL."
    }
    else {
        Write-Host "📦 Кэш модели Whisper large-v3 не найден. Скачиваем модель с huggingface"
        $PythonLoadScript = @"
import os
os.environ['HF_HOME'] = r'$ModelCacheWinWsl'
from faster_whisper import WhisperModel

print('5.5.2 (python) 🔄 Кэшируем Whisper large-v3 на CPU...')
WhisperModel('large-v3', device='cpu', compute_type='int8')
print('5.5.2 (python) ✅ Модель закэширована для CPU.')

try:
    print('5.5.2 (python) 🔄 Кэшируем Whisper large-v3 на GPU (если поддерживается)...')
    WhisperModel('large-v3', device='cuda', compute_type='int8_float16')
    print('5.5.2 (python) ✅ Модель закэширована для GPU.')
except Exception as e:
    print('5.5.2 (python) ⚠️ Не удалось закэшировать для GPU: ' + str(e))
"@
        $TempPreload = Join-Path $TempDir "preload_large_v3.py"
        $PythonLoadScript | Out-File -FilePath $TempPreload -Encoding UTF8
        $TempPreloadWsl = Convert-WindowsPathToWsl $TempPreload
		#Отладка
		Write-Host "`n🔎 Отладка: путь скрипта preload внутри WSL: $TempPreloadWsl"
		#
        wsl -d $DistroName -- bash -c "python3 '$TempPreloadWsl'"
        Remove-Item $TempPreload -Force

        Write-Host "📦 Кэш модели Whisper large-v3 скачен. Копируем в WSL..."
        wsl -d $DistroName -- bash -c "mkdir -p '$ModelCacheLocalWsl' && cp -r '$ModelCacheWinWsl/hub/'* '$ModelCacheLocalWsl/'"
        Write-Host "✅ Кэш успешно скачен и скопирован Windows => WSL."
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

#Write-Host "❌ СТОП ТЕСТ"
#exit 1