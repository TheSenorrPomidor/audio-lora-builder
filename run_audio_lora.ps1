# === run_audio_lora.ps1 ===
chcp 65001 > $null
$OutputEncoding = [System.Text.UTF8Encoding]::new()
Write-Host "🚀 Копируем и запускаем process_audio.py в WSL..."

# 1. Пути
$Distro = "audio-lora"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$TargetLinuxPath = "/root/audio-lora-builder"

# 2. Список файлов для копирования
$FilesToCopy = @(
    "process_audio.py",
    "speaker_embedding_db.py"
)

# 3. Создаём каталог в WSL
wsl -d $Distro -- mkdir -p $TargetLinuxPath

foreach ($file in $FilesToCopy) {
    $ScriptPath = Join-Path $ScriptDir $file
    $TargetLinuxFile = "$TargetLinuxPath/$file"
    
    # Удаляем старую версию
    wsl -d $Distro -- rm -f $TargetLinuxFile

    # Копируем через echo
    if (Test-Path $ScriptPath) {
        Write-Host "📄 Копируем: $file"
        Get-Content $ScriptPath -Raw | wsl -d $Distro -- bash -c "cat > $TargetLinuxFile"
    } else {
        Write-Warning "⚠️ Не найден файл: $file"
    }
}

# 4. Запускаем
Write-Host "▶️ Запускаем process_audio.py в WSL..."
wsl -d $Distro -- python3 "$TargetLinuxPath/process_audio.py"

Write-Host "`n✅ Выполнение run_audio_lora.ps1 завершено."
