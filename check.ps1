function Convert-WindowsPathToWsl {
    param (
        [Parameter(Mandatory=$true)]
        [string]$WindowsPath
    )
    $WslPath = $WindowsPath -replace '\\', '/' -replace '^([A-Za-z]):', '/mnt/$1'
    return $WslPath.ToLower()
}
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$TempDir = Join-Path $ScriptDir "temp"
$DistroName = "audio-lora"



$CudaKeyUrl     = "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-archive-keyring.gpg"
$LocalCudaDebs = Join-Path $TempDir "cuda_debs"
$WslCudaDebs    = Convert-WindowsPathToWsl $LocalCudaDebs
$CudaKeyFile    = Join-Path $LocalCudaDebs "cuda-archive-keyring.gpg"
$WslCudaKeyFile = Convert-WindowsPathToWsl $CudaKeyFile

wsl -d audio-lora -- bash -c "echo 'nameserver 8.8.8.8' > /etc/resolv.conf"

#Отключаем официальные репо, что бы ставить только оффлайн
wsl -d $DistroName -- bash -c "sudo mv /etc/apt/sources.list /etc/apt/sources.list.bak"

#Добавляем репозиторий CUDA (signed-by)
wsl -d $DistroName -- bash -c "echo 'deb [signed-by=$WslCudaKeyFile] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /' | sudo tee /etc/apt/sources.list.d/cuda.list"



############################


############################





<#
# === 5.4 Установка Python-библиотек (torch, whisperx или faster-whisper) ===
Write-Host "`n5.4 📦 Установка Python-библиотек (torch, $WhisperImpl) из temp\pip (Windows)"
$PipCacheWin = Join-Path $TempDir "pip"
$PipCacheWsl = Convert-WindowsPathToWsl $PipCacheWin

# если uv не установлен — ставим
$UvCheck = wsl -d $DistroName -- bash -c "command -v uv || echo no"
if ($UvCheck -eq "no") {
	Write-Host "📦 Установка uv..."
	wsl -d $DistroName -- bash -c "curl -LsSf https://astral.sh/uv/install.sh | sh"
}

# путь к .in/.txt
$ReqIn  = "requirements_pypi.in"
$ReqTxt = "requirements_pypi.txt"
$ReqInFull  = Join-Path $ProjectRoot $ReqIn
$ReqTxtFull = Join-Path $ProjectRoot $ReqTxt

# генерируем .in если его нет
if (-not (Test-Path $ReqInFull)) {
	Write-Host "🧾 Не найден $ReqIn — генерируем из списка..."
	@(
		"faster-whisper"
	) | Set-Content -Encoding UTF8 $ReqInFull
}

# генерируем .txt если его нет
if (-not (Test-Path $ReqTxtFull)) {
	Write-Host "📋 Компилируем $ReqTxt из $ReqIn..."
	wsl -d $DistroName -- bash -c "cd '$ProjectRootWsl' && uv pip compile $ReqIn --extra-index-url https://download.pytorch.org/whl/cu118 --output-file $ReqTxt"
}

# создаём temp/pip если нужно
if (-not (Test-Path $PipCacheWin)) {
	New-Item -ItemType Directory -Path $PipCacheWin | Out-Null
}

# скачиваем .whl
Write-Host "⬇️ Скачиваем пакеты из $ReqTxt..."
wsl -d $DistroName -- bash -c "cd '$ProjectRootWsl' && pip download -r $ReqTxt -d '$PipCacheWsl'"

# устанавливаем из .whl
Write-Host "📦 Устанавливаем Python-библиотеки (по $ReqTxt)..."
wsl -d $DistroName -- bash -c "cd '$ProjectRootWsl' && pip install --no-index --find-links='$PipCacheWsl' -r $ReqTxt"





давай рабтать над твоим патчем:
1. $PyWheels ты не используешь, хотя у нас он есть и Source что бы определить к какому списку его отнести pypi или torch  и Impl есть что бы понять нужно ли нам сейчас ставить фастер виспер или висперХ и Name для чёткого понимания имени пакета
2. Не вижу что бы мы генерировали файлы по $PyWheels
3. В оригинальном скрипте мы првоеряем установлен ли пакет     $Files = Get-ChildItem -Path $PipCacheWin -Filter $pkg.Pattern. При этом я не уверен что Pattern  даст точный ответ о версии пакета, логичней проверять пакет с его версией(!) силами pip. В твоем варианте я вообще проверок не вижу.

Я предлагаю тебе полностью построчно перебрать "5.4 Установка Python-библиотек..." и где возможно внедрить новую доработанную структуру установщика whl



---------------------------------------------------




# === 5.4 Установка Python-библиотек (из .whl) ===
Write-Section "5.4 Установка Python-библиотек (.whl)"

# Устанавливаем uv, если не установлен
if (-not (Test-Path "$VEnvPath/bin/uv")) {
    Write-Step "Установка uv..."
    & "$VEnvPath/bin/pip" install uv
}

# Папки и пути
$ReqTorchIn  = "$PSScriptRoot/requirements_pytorch.in"
$ReqTorchTxt = "$PSScriptRoot/requirements_pytorch.txt"
$ReqPypiIn   = "$PSScriptRoot/requirements_pypi.in"
$ReqPypiTxt  = "$PSScriptRoot/requirements_pypi.txt"

# 1. Генерация .in из $PyWheels при отсутствии
if (!(Test-Path $ReqTorchIn)) {
    Write-Step "Генерация requirements_pytorch.in..."
    $Lines = $PyWheels | Where-Object { $_.Source -eq "torch" -and $_.Impl -ne "skip" } | ForEach-Object { $_.Name }
    $Lines | Set-Content -Encoding UTF8 $ReqTorchIn
}
if (!(Test-Path $ReqPypiIn)) {
    Write-Step "Генерация requirements_pypi.in..."
    $Lines = $PyWheels | Where-Object { $_.Source -eq "pypi" -and $_.Impl -ne "skip" } | ForEach-Object { $_.Name }
    $Lines | Set-Content -Encoding UTF8 $ReqPypiIn
}

# 2. Генерация .txt из .in (если нужно)
if (!(Test-Path $ReqTorchTxt)) {
    Write-Step "Компиляция requirements_pytorch.txt через uv..."
    & "$VEnvPath/bin/uv" pip compile $ReqTorchIn --extra-index-url https://download.pytorch.org/whl/cu118 --output-file $ReqTorchTxt
}
if (!(Test-Path $ReqPypiTxt)) {
    Write-Step "Компиляция requirements_pypi.txt через uv..."
    & "$VEnvPath/bin/uv" pip compile $ReqPypiIn --output-file $ReqPypiTxt
}

# 3. Установка .whl
function Install-WheelSet {
    param (
        [string]$Label,
        [string]$TxtPath
    )
    if (Test-Path $TxtPath) {
        Write-Step "Установка Python-библиотек $Label (offline)..."
        & "$VEnvPath/bin/pip" install --no-index --find-links="$PipCacheWin" -r $TxtPath
    } else {
        Write-Warning "Пропущено: $Label — нет $TxtPath"
    }
}

Install-WheelSet "из PyTorch" $ReqTorchTxt
Install-WheelSet "из PyPI"    $ReqPypiTxt



"# Устанавливаем uv, если не установлен
if (-not (Test-Path "$VEnvPath/bin/uv")) {
    Write-Step "Установка uv..."
    & "$VEnvPath/bin/pip" install uv
}"
Сначала надо проверить установлены ли необходимые пакеты в линуксе, ведь если всё стоит надо просто проскипать блок установки (у нас так в текущем файле и это логично)

если установленные пакеты не нашли - проверить скачены ли ключевые whl (у нас так в текущем файле и это логично)

И если скаченные файлы отсутствуют хотя бы частично, устанавливаем pip install uv и генерируем requirements.in/txt по  $PyWheels в рамках выбранного Impl
этого блока вообще нет, он полностью новый

<= давай дойдём до сюда сначала =>
далее будет модификация блока "#Скачиваем отсутствующие пакеты .whl в temp/pip". Теперь мы качаем через requirements_pytorch и requirements_pypi

далее аналогичная модификация "#Устанавливаем из temp/pip"




------------------------------------------------------------------------------------------




А мне кажется наш имеющийся блок проверки хорош, надо только улучшить (наверное) 
		$DepName = $pkg.Name -split '==|\+' | Select-Object -First 1
		$IsInstalled = wsl -d $DistroName -- bash -c "pip show $DepName > /dev/null && echo ok"
У меня есть подозрения что он не проверяет конкретную версию установленного pip пакета whl. Особенно эта сумрачная часть "-split '==|\+' | Select-Object -First 1" модифицирует $pkg.Name не понятно во что, оставляет только имя "torch==2.0.0+cu118 => torch"? Если так то это туматч, нам надо управлять конкретными версиями
		
Текущий блок :"		
$PyWheelsMissing = @()
foreach ($pkg in $PyWheels) {
    $IsForThisImpl = ($pkg.Impl -eq "all" -or $pkg.Impl -eq $WhisperImpl)
    if ($IsForThisImpl) {
		# Выделяем имя пакета без версии
		$DepName = $pkg.Name -split '==|\+' | Select-Object -First 1 
		$DepVersion = $pkg.Name -split '==|\+' | Select-Object -First 2 <= заменить на правильную регулярку
		$IsInstalled = wsl -d $DistroName -- bash -c "pip show $DepName > /dev/null && echo ok"
		if ($IsInstalled -eq "ok") {
			Write-Host "✅ $($pkg.Name) уже установлен — пропускаем"
		} else {
			Write-Host "⬇️ $($pkg.Name) не установлен — добавляем в обработку"
			$PyWheelsMissing += $pkg
		}
	}
}
"

Пока этот блок не вылечим не иди дальше

----------------------------
Наведём порядок, а то мы часть регулярок вывели в $DepVersion, а часть прямо в wsl команде

$DepName = $pkg.Name -split '==|\+' | Select-Object -First 1 
$DepVersion = $pkg.Name -split '==|\+' | Select-Object -First 2 <= заменить на правильную регулярку

$IsInstalled = wsl -d $DistroName -- bash -c "pip show $DepName 2>/dev/null | grep -q 'Version: $DepVersion' && echo ok" <= это надо починить, я схематично накидал псевдо код, так же надо что бы мы понимали что пакет установлен но версия не та, нам же надо будет понимать что пакеты надо удалить, ага?


--------------------------



$DepName = $pkg.Name -split '==|\+' | Select-Object -First 1
$DepVersion = $pkg.Name -split '==|\+' | Select-Object -Skip 1 | Select-Object -First 1
$InstalledVersion = wsl -d $DistroName -- bash -c "pip show $DepName 2>/dev/null | grep '^Version:' | awk '{print \$2}'"

if ($InstalledVersion -eq $DepVersion) {
	Write-Host "✅ $($pkg.Name) уже установлен — пропускаем"
} elseif ($InstalledVersion) {
	Write-Host "⚠️ $DepName установлен, но версия $InstalledVersion ≠ $DepVersion — добавляем в обработку"
	$PyWheelsMissing += $pkg
} else {
	Write-Host "⬇️ $($pkg.Name) не установлен — добавляем в обработку"
	$PyWheelsMissing += $pkg
}











#>
