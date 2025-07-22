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



	Write-Host "❌ СТОП ТЕСТ"; exit 1
<#
PS D:\VM\WSL2\audio-lora-builder> .\install_audio_lora.ps1

Версия скрипта install_audio_lora.ps1 4.2

1. 🔍 Проверяем наличие WSL-дистрибутива 'audio-lora'...

2. 📦 Поиск базового или финального rootfs

✅ Найден финальный rootfs: D:\VM\WSL2\audio-lora-builder\rootfs\audio_lora_rootfs.tar.gz

3. 💽 Импортируем WSL-дистрибутив из 'D:\VM\WSL2\audio-lora-builder\rootfs\audio_lora_rootfs.tar.gz'...
Операция успешно завершена.

4. 🌐 Восстанавливаем DNS в WSL...
5. 📦 === Установка зависимостей ===

5.1 📦 Установка python3-pip, ffmpeg, dpkg-dev, unzip
✅ python3-pip уже установлен, пропускаем.
✅ ffmpeg уже установлен, пропускаем.
✅ dpkg-dev уже установлен, пропускаем.
✅ unzip уже установлен, пропускаем.
📦 Все пакеты уже установлены. Установка не требуется.

5.2 📦 Установка CUDA Runtime 12.6 (через apt-get --download-only, оффлайн) v10
✅ libcublas-12-6 уже установлен, пропускаем.
✅ libcublas-dev-12-6 уже установлен, пропускаем.
✅ cuda-toolkit-12-config-common уже установлен, пропускаем.
✅ cuda-toolkit-12-6-config-common уже установлен, пропускаем.
✅ cuda-toolkit-config-common уже установлен, пропускаем.
✅ cuda-runtime-12-6 уже установлен, пропускаем.
✅ Весь CUDA Runtime уже установлен. Пропускаем установку CUDA.
5.3 📦 Установка cuDNN
✅ cuDNN уже установлен в WSL — пропускаем установку.

5.4 📦 Установка Python-библиотек (torch, whisperx) из temp\pip (Windows)
✅ whisperx==3.3.1 уже установлен — пропускаем
✅ transformers==4.28.1 уже установлен — пропускаем
✅ librosa==0.10.0 уже установлен — пропускаем
✅ Все необходимые Python-библиотеки установлены

5.5 🧠 Предзагрузка и кэширование модели whisperx для CPU и GPU
📦 Определение переменных
📦 Проверка установленного кэша модели WhisperX large-v3 в WSL...
📦 Кэш в WSL не найден, ищем кэш в temp: D:\VM\WSL2\audio-lora-builder\temp\huggingface\whisperx
📦 Кэш модели WhisperX large-v3 не найден. Скачиваем модель с huggingface
(python) 🔄 Кэшируем WhisperX large-v3 на CPU...
preprocessor_config.json: 100%|████████████████████████████████████████████████████████| 340/340 [00:00<00:00, 4.62MB/s]
config.json: 2.39kB [00:00, 18.0MB/s]                                                       | 0.00/3.09G [00:00<?, ?B/s]
vocabulary.json: 1.07MB [00:00, 9.46MB/s]
tokenizer.json: 2.48MB [00:00, 12.5MB/s]]
model.bin: 100%|███████████████████████████████████████████████████████████████████| 3.09G/3.09G [06:25<00:00, 8.01MB/s]
Traceback (most recent call last):7MB/s]
  File "/mnt/d/vm/wsl2/audio-lora-builder/temp/preload_whisperx.py", line 7, in <module>
    model = whisperx.load_model("large-v3", device="cpu")
  File "/usr/local/lib/python3.10/dist-packages/whisperx/asr.py", line 325, in load_model
    model = model or WhisperModel(whisper_arch,
  File "/usr/local/lib/python3.10/dist-packages/faster_whisper/transcribe.py", line 634, in __init__
    self.model = ctranslate2.models.Whisper(
ValueError: Requested float16 compute type, but the target device or backend do not support efficient float16 computation.
📦 Кэш модели WhisperX large-v3 скачен. Копируем в WSL...
✅ Кэш скачен и скопирован Windows => WSL.








5.5 🧠 Предзагрузка и кэширование модели whisperx для CPU и GPU
📦 Определение переменных
📦 Проверка установленного кэша модели WhisperX large-v3 в WSL...
find: ‘/root/.cache/huggingface/hub’: No such file or directory
📦 Кэш в WSL не найден, ищем кэш в temp: D:\VM\WSL2\audio-lora-builder\temp\huggingface\whisperx
📦 Кэш модели WhisperX large-v3 не найден. Скачиваем модель с huggingface
(python) 🔄 Кэшируем WhisperX large-v3 на CPU...
Traceback (most recent call last):
  File "/mnt/d/vm/wsl2/audio-lora-builder/temp/preload_whisperx.py", line 7, in <module>
    model = whisperx.load_model("large-v3", device="cpu")
  File "/usr/local/lib/python3.10/dist-packages/whisperx/asr.py", line 325, in load_model
    model = model or WhisperModel(whisper_arch,
  File "/usr/local/lib/python3.10/dist-packages/faster_whisper/transcribe.py", line 634, in __init__
    self.model = ctranslate2.models.Whisper(
ValueError: Requested float16 compute type, but the target device or backend do not support efficient float16 computation.



python3 -c "import whisperx; whisperx.load_model('large-v3', device='cpu', compute_type='int8')"

python3 -c "import whisperx; whisperx.load_model('large-v3', device='cuda', compute_type='int8_float16')"



#>





