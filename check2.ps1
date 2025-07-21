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
			wsl -d $DistroName -- bash -c "pip download -r '$txtPathWsl' -d '$PipCacheWsl'"
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














1. Может же в разное время утилита UV дать разный набор зависимостей/версий для одного и тогоже пакета, например  whisperx==3.3.1
2. Требуется ли UV для генерации txt файла интернет?
Если по п.1 и п.2 ответ ДА, то предлагаю:
Если проверка скаченных пакетов в папке temp/pip показала что все файлы есть И файл TXT есть, то заново файл TXT не генерим










Тестовый прогон:
Есть ошибки, при этом я обратил внимание что в папке temp/pip только 3 файла:
requirements_torch.in
requirements_torch.txt
uv-0.8.0-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
Хотя роде что то много много качалось минимум 2.5Гб
Как будто что то качалось в линукс WSL во временную папку и оттуда уже не вышло (возмоно и от прошлых попыток закачки, когда мы отрабатывали скрипт)



5.4 📦 Установка Python-библиотек (torch, whisperx) из temp\pip (Windows)
WARNING: Package(s) not found: whisperx
⬇️ whisperx==3.3.1 не установлен — добавляем в обработку
WARNING: Package(s) not found: transformers
⬇️ transformers==4.28.1 не установлен — добавляем в обработку
WARNING: Package(s) not found: librosa
⬇️ librosa==0.10.0 не установлен — добавляем в обработку
📦 Чтение .whl файлов завершено.
⬇️ В temp/pip whisperx==3.3.1 не найден
⬇️ В temp/pip transformers==4.28.1 не найден
⬇️ В temp/pip librosa==0.10.0 не найден
Collecting uv
  Using cached uv-0.8.0-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (18.7 MB)
Saved /mnt/d/vm/wsl2/audio-lora-builder/temp/pip/uv-0.8.0-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
Successfully downloaded uv

📄 Генерация torch requirements...
🌐 Загрузка зависимостей torch...
Collecting aiohappyeyeballs==2.6.1
  Using cached aiohappyeyeballs-2.6.1-py3-none-any.whl (15 kB)
Collecting aiohttp==3.12.14
  Using cached aiohttp-3.12.14-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (1.6 MB)
Collecting aiosignal==1.4.0
  Using cached aiosignal-1.4.0-py3-none-any.whl (7.5 kB)
Collecting alembic==1.16.4
  Using cached alembic-1.16.4-py3-none-any.whl (247 kB)
Collecting antlr4-python3-runtime==4.9.3
  Using cached antlr4-python3-runtime-4.9.3.tar.gz (117 kB)
  Preparing metadata (setup.py) ... done
Collecting asteroid-filterbanks==0.4.0
  Using cached asteroid_filterbanks-0.4.0-py3-none-any.whl (29 kB)
Collecting async-timeout==5.0.1
  Using cached async_timeout-5.0.1-py3-none-any.whl (6.2 kB)
Collecting attrs==25.3.0
  Using cached attrs-25.3.0-py3-none-any.whl (63 kB)
Collecting audioread==3.0.1
  Using cached audioread-3.0.1-py3-none-any.whl (23 kB)
Collecting av==15.0.0
  Using cached av-15.0.0-cp310-cp310-manylinux_2_28_x86_64.whl (39.2 MB)
Collecting certifi==2022.12.7
  Using cached certifi-2022.12.7-py3-none-any.whl (155 kB)
Collecting cffi==1.17.1
  Using cached cffi-1.17.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (446 kB)
Collecting charset-normalizer==2.1.1
  Using cached charset_normalizer-2.1.1-py3-none-any.whl (39 kB)
Collecting click==8.2.1
  Using cached click-8.2.1-py3-none-any.whl (102 kB)
Collecting coloredlogs==15.0.1
  Using cached coloredlogs-15.0.1-py2.py3-none-any.whl (46 kB)
Collecting colorlog==6.9.0
  Using cached colorlog-6.9.0-py3-none-any.whl (11 kB)
Collecting contourpy==1.3.2
  Using cached contourpy-1.3.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (325 kB)
Collecting ctranslate2==4.4.0
  Using cached ctranslate2-4.4.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (37.2 MB)
Collecting cycler==0.12.1
  Using cached cycler-0.12.1-py3-none-any.whl (8.3 kB)
Collecting decorator==5.2.1
  Using cached decorator-5.2.1-py3-none-any.whl (9.2 kB)
Collecting docopt==0.6.2
  Using cached docopt-0.6.2.tar.gz (25 kB)
  Preparing metadata (setup.py) ... done
Collecting einops==0.8.1
  Using cached einops-0.8.1-py3-none-any.whl (64 kB)
Collecting faster-whisper==1.1.0
  Using cached faster_whisper-1.1.0-py3-none-any.whl (1.1 MB)
Collecting filelock==3.13.1
  Using cached filelock-3.13.1-py3-none-any.whl (11 kB)
Collecting flatbuffers==25.2.10
  Using cached flatbuffers-25.2.10-py2.py3-none-any.whl (30 kB)
Collecting fonttools==4.59.0
  Using cached fonttools-4.59.0-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (4.8 MB)
Collecting frozenlist==1.7.0
  Using cached frozenlist-1.7.0-cp310-cp310-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl (222 kB)
Collecting fsspec==2024.6.1
  Using cached fsspec-2024.6.1-py3-none-any.whl (177 kB)
Collecting greenlet==3.2.3
  Using cached greenlet-3.2.3-cp310-cp310-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl (582 kB)
Collecting hf-xet==1.1.5
  Using cached hf_xet-1.1.5-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (3.1 MB)
Collecting huggingface-hub==0.33.4
  Using cached huggingface_hub-0.33.4-py3-none-any.whl (515 kB)
Collecting humanfriendly==10.0
  Using cached humanfriendly-10.0-py2.py3-none-any.whl (86 kB)
Collecting hyperpyyaml==1.2.2
  Using cached HyperPyYAML-1.2.2-py3-none-any.whl (16 kB)
Collecting idna==3.4
  Using cached idna-3.4-py3-none-any.whl (61 kB)
Collecting jinja2==3.1.4
  Using cached jinja2-3.1.4-py3-none-any.whl (133 kB)
Collecting joblib==1.5.1
  Using cached joblib-1.5.1-py3-none-any.whl (307 kB)
Collecting julius==0.2.7
  Using cached julius-0.2.7.tar.gz (59 kB)
  Preparing metadata (setup.py) ... done
Collecting kiwisolver==1.4.8
  Using cached kiwisolver-1.4.8-cp310-cp310-manylinux_2_12_x86_64.manylinux2010_x86_64.whl (1.6 MB)
Collecting lazy-loader==0.4
  Using cached lazy_loader-0.4-py3-none-any.whl (12 kB)
Collecting librosa==0.10.0
  Using cached librosa-0.10.0-py3-none-any.whl (252 kB)
Collecting lightning==2.5.2
  Using cached lightning-2.5.2-py3-none-any.whl (821 kB)
Collecting lightning-utilities==0.11.8
  Using cached lightning_utilities-0.11.8-py3-none-any.whl (26 kB)
Collecting llvmlite==0.44.0
  Using cached llvmlite-0.44.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (42.4 MB)
Collecting mako==1.3.10
  Using cached mako-1.3.10-py3-none-any.whl (78 kB)
Collecting markdown-it-py==3.0.0
  Using cached markdown_it_py-3.0.0-py3-none-any.whl (87 kB)
Collecting markupsafe==2.1.5
  Using cached MarkupSafe-2.1.5-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (25 kB)
Collecting matplotlib==3.10.3
  Using cached matplotlib-3.10.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (8.6 MB)
Collecting mdurl==0.1.2
  Using cached mdurl-0.1.2-py3-none-any.whl (10.0 kB)
Collecting mpmath==1.3.0
  Using cached mpmath-1.3.0-py3-none-any.whl (536 kB)
Collecting msgpack==1.1.1
  Using cached msgpack-1.1.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (408 kB)
Collecting multidict==6.6.3
  Using cached multidict-6.6.3-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl (241 kB)
Collecting networkx==3.3
  Using cached networkx-3.3-py3-none-any.whl (1.7 MB)
Collecting nltk==3.9.1
  Using cached nltk-3.9.1-py3-none-any.whl (1.5 MB)
Collecting numba==0.61.2
  Using cached numba-0.61.2-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (3.8 MB)
Collecting numpy==2.1.2
  Using cached numpy-2.1.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (16.3 MB)
Collecting nvidia-cublas-cu11==11.11.3.6
  Downloading nvidia_cublas_cu11-11.11.3.6-py3-none-manylinux2014_x86_64.whl (417.9 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 417.9/417.9 MB 2.0 MB/s eta 0:00:00
Collecting nvidia-cuda-cupti-cu11==11.8.87
  Downloading nvidia_cuda_cupti_cu11-11.8.87-py3-none-manylinux2014_x86_64.whl (13.1 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 13.1/13.1 MB 3.8 MB/s eta 0:00:00
Collecting nvidia-cuda-nvrtc-cu11==11.8.89
  Downloading nvidia_cuda_nvrtc_cu11-11.8.89-py3-none-manylinux2014_x86_64.whl (23.2 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 23.2/23.2 MB 2.6 MB/s eta 0:00:00
Collecting nvidia-cuda-runtime-cu11==11.8.89
  Downloading nvidia_cuda_runtime_cu11-11.8.89-py3-none-manylinux2014_x86_64.whl (875 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 875.6/875.6 KB 2.3 MB/s eta 0:00:00
Collecting nvidia-cudnn-cu11==9.1.0.70
  Downloading nvidia_cudnn_cu11-9.1.0.70-py3-none-manylinux2014_x86_64.whl (663.9 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 663.9/663.9 MB 915.8 kB/s eta 0:00:00
Collecting nvidia-cufft-cu11==10.9.0.58
  Downloading nvidia_cufft_cu11-10.9.0.58-py3-none-manylinux2014_x86_64.whl (168.4 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 168.4/168.4 MB 1.5 MB/s eta 0:00:00
Collecting nvidia-curand-cu11==10.3.0.86
  Downloading nvidia_curand_cu11-10.3.0.86-py3-none-manylinux2014_x86_64.whl (58.1 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 58.1/58.1 MB 1.4 MB/s eta 0:00:00
Collecting nvidia-cusolver-cu11==11.4.1.48
  Downloading nvidia_cusolver_cu11-11.4.1.48-py3-none-manylinux2014_x86_64.whl (128.2 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 128.2/128.2 MB 1.2 MB/s eta 0:00:00
Collecting nvidia-cusparse-cu11==11.7.5.86
  Downloading nvidia_cusparse_cu11-11.7.5.86-py3-none-manylinux2014_x86_64.whl (204.1 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 204.1/204.1 MB 3.2 MB/s eta 0:00:00
Collecting nvidia-nccl-cu11==2.21.5
  Downloading nvidia_nccl_cu11-2.21.5-py3-none-manylinux2014_x86_64.whl (147.8 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 147.8/147.8 MB 1.6 MB/s eta 0:00:00
Collecting nvidia-nvtx-cu11==11.8.86
  Downloading nvidia_nvtx_cu11-11.8.86-py3-none-manylinux2014_x86_64.whl (99 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 99.1/99.1 KB 614.2 kB/s eta 0:00:00
Collecting omegaconf==2.3.0
  Using cached omegaconf-2.3.0-py3-none-any.whl (79 kB)
Collecting onnxruntime==1.22.1
  Using cached onnxruntime-1.22.1-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (16.5 MB)
Collecting optuna==4.4.0
  Using cached optuna-4.4.0-py3-none-any.whl (395 kB)
Collecting packaging==24.1
  Downloading packaging-24.1-py3-none-any.whl (53 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 54.0/54.0 KB 471.6 kB/s eta 0:00:00
Collecting pandas==2.3.1
  Using cached pandas-2.3.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (12.3 MB)
Collecting pillow==11.0.0
  Downloading pillow-11.0.0-cp310-cp310-manylinux_2_28_x86_64.whl (4.4 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 4.4/4.4 MB 1.5 MB/s eta 0:00:00
Collecting platformdirs==4.3.8
  Using cached platformdirs-4.3.8-py3-none-any.whl (18 kB)
Collecting pooch==1.8.2
  Using cached pooch-1.8.2-py3-none-any.whl (64 kB)
Collecting primepy==1.3
  Using cached primePy-1.3-py3-none-any.whl (4.0 kB)
Collecting propcache==0.3.2
  Using cached propcache-0.3.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (198 kB)
Collecting protobuf==6.31.1
  Using cached protobuf-6.31.1-cp39-abi3-manylinux2014_x86_64.whl (321 kB)
Collecting pyannote-audio==3.3.2
  Using cached pyannote.audio-3.3.2-py2.py3-none-any.whl (898 kB)
Collecting pyannote-core==5.0.0
  Using cached pyannote.core-5.0.0-py3-none-any.whl (58 kB)
Collecting pyannote-database==5.1.3
  Using cached pyannote.database-5.1.3-py3-none-any.whl (48 kB)
Collecting pyannote-metrics==3.2.1
  Using cached pyannote.metrics-3.2.1-py3-none-any.whl (51 kB)
Collecting pyannote-pipeline==3.0.1
  Using cached pyannote.pipeline-3.0.1-py3-none-any.whl (31 kB)
Collecting pycparser==2.22
  Using cached pycparser-2.22-py3-none-any.whl (117 kB)
Collecting pygments==2.19.2
  Using cached pygments-2.19.2-py3-none-any.whl (1.2 MB)
Collecting pyparsing==3.2.3
  Using cached pyparsing-3.2.3-py3-none-any.whl (111 kB)
Collecting python-dateutil==2.9.0.post0
  Using cached python_dateutil-2.9.0.post0-py2.py3-none-any.whl (229 kB)
Collecting pytorch-lightning==2.5.2
  Using cached pytorch_lightning-2.5.2-py3-none-any.whl (825 kB)
Collecting pytorch-metric-learning==2.8.1
  Using cached pytorch_metric_learning-2.8.1-py3-none-any.whl (125 kB)
Collecting pytz==2025.2
  Using cached pytz-2025.2-py2.py3-none-any.whl (509 kB)
Collecting pyyaml==6.0.2
  Using cached PyYAML-6.0.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (751 kB)
Collecting regex==2024.11.6
  Using cached regex-2024.11.6-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (781 kB)
Collecting requests==2.28.1
  Downloading requests-2.28.1-py3-none-any.whl (62 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 62.8/62.8 KB 522.4 kB/s eta 0:00:00
Collecting rich==14.0.0
  Using cached rich-14.0.0-py3-none-any.whl (243 kB)
Collecting ruamel-yaml==0.18.14
  Using cached ruamel.yaml-0.18.14-py3-none-any.whl (118 kB)
Collecting ruamel-yaml-clib==0.2.12
  Using cached ruamel.yaml.clib-0.2.12-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (722 kB)
Collecting scikit-learn==1.7.1
  Downloading scikit_learn-1.7.1-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (9.7 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 9.7/9.7 MB 1.7 MB/s eta 0:00:00
Collecting scipy==1.15.3
  Using cached scipy-1.15.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (37.7 MB)
Collecting semver==3.0.4
  Using cached semver-3.0.4-py3-none-any.whl (17 kB)
Collecting sentencepiece==0.2.0
  Using cached sentencepiece-0.2.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (1.3 MB)
Collecting setuptools==70.2.0
  Downloading setuptools-70.2.0-py3-none-any.whl (930 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 930.8/930.8 KB 853.3 kB/s eta 0:00:00
Collecting shellingham==1.5.4
  Using cached shellingham-1.5.4-py2.py3-none-any.whl (9.8 kB)
Collecting six==1.17.0
  Using cached six-1.17.0-py2.py3-none-any.whl (11 kB)
Collecting sortedcontainers==2.4.0
  Using cached sortedcontainers-2.4.0-py2.py3-none-any.whl (29 kB)
Collecting soundfile==0.13.1
  Using cached soundfile-0.13.1-py2.py3-none-manylinux_2_28_x86_64.whl (1.3 MB)
Collecting soxr==0.5.0.post1
  Using cached soxr-0.5.0.post1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (252 kB)
Collecting speechbrain==1.0.3
  Using cached speechbrain-1.0.3-py3-none-any.whl (864 kB)
Collecting sqlalchemy==2.0.41
  Using cached sqlalchemy-2.0.41-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (3.2 MB)
Collecting sympy==1.13.3
  Using cached sympy-1.13.3-py3-none-any.whl (6.2 MB)
Collecting tabulate==0.9.0
  Using cached tabulate-0.9.0-py3-none-any.whl (35 kB)
Collecting tensorboardx==2.6.4
  Using cached tensorboardx-2.6.4-py3-none-any.whl (87 kB)
Collecting threadpoolctl==3.6.0
  Using cached threadpoolctl-3.6.0-py3-none-any.whl (18 kB)
Collecting tokenizers==0.13.3
  Using cached tokenizers-0.13.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (7.8 MB)
Collecting tomli==2.2.1
  Using cached tomli-2.2.1-py3-none-any.whl (14 kB)
ERROR: Could not find a version that satisfies the requirement torch==2.7.1+cu118 (from versions: 1.11.0, 1.12.0, 1.12.1, 1.13.0, 1.13.1, 2.0.0, 2.0.1, 2.1.0, 2.1.1, 2.1.2, 2.2.0, 2.2.1, 2.2.2, 2.3.0, 2.3.1, 2.4.0, 2.4.1, 2.5.0, 2.5.1, 2.6.0, 2.7.0, 2.7.1)
ERROR: No matching distribution found for torch==2.7.1+cu118
📦 Установка torch-пакетов...
Looking in links: /mnt/d/vm/wsl2/audio-lora-builder/temp/pip
ERROR: Could not find a version that satisfies the requirement aiohappyeyeballs==2.6.1 (from versions: none)
ERROR: No matching distribution found for aiohappyeyeballs==2.6.1
WARNING: Package(s) not found: whisperx
❌ whisperx==3.3.1 установить не удалось
WARNING: Package(s) not found: transformers
❌ transformers==4.28.1 установить не удалось
WARNING: Package(s) not found: librosa
❌ librosa==0.10.0 установить не удалось
❌ СТОП ТЕСТ







Правим только этот блок:
    Write-Host "`n📄 Генерация $group requirements..."
    wsl -d $DistroName -- bash -c "$compileCmd > /dev/null 2>&1"
	
Мой алгоритм который ты проигнорировал: "Если проверка скаченных пакетов в папке temp/pip показала что все файлы есть И файл TXT есть, то заново файл TXT не генерим"	
Код должен получиться типо такого:
Если (файл тхт есть И $PyWheelsToDownload.Количество (x=> x.Source = $group) = 0)
    Write-Host "`n📄 Все ключевые блоки по группе $group скачены и файл group_тхт уже есть, Генерация тхт $group не требуется..."

Иначе 
    Write-Host "`n📄 Генерация $group requirements..."
    wsl -d $DistroName -- bash -c "$compileCmd > /dev/null 2>&1"



    # Проверяем, если .txt файл уже существует, пропускаем генерацию
    if (Test-Path $txtPathWin) {
        Write-Host "📄 $group requirements.txt уже существует, пропускаем генерацию."
        return
    }
#>

