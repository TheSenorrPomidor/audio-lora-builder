# === run_audio_lora.ps1 ===
chcp 65001 > $null
$OutputEncoding = [System.Text.UTF8Encoding]::new()
Write-Host "`n🚀 Копируем и запускаем process_audio.py в WSL..."

# 1. Пути
$Distro = "audio-lora"
$ScriptPath = Join-Path $PSScriptRoot "process_audio.py"
$TargetLinuxPath = "/root/audio-lora-builder"
$TargetLinuxFile = "$TargetLinuxPath/process_audio.py"

# 2. Создаём каталог в WSL
wsl -d $Distro -- mkdir -p $TargetLinuxPath
wsl -d $Distro -- rm -f $TargetLinuxFile

# 3. Читаем содержимое Python-файла и копируем в WSL через echo в bash
Get-Content $ScriptPath -Raw | wsl -d $Distro -- bash -c "cat > $TargetLinuxFile"

# 4. Запускаем
Write-Host "`n▶️ Запускаем process_audio.py в WSL..."
wsl -d $Distro -- python3 $TargetLinuxFile

Write-Host "`n✅ Выполнение завершено."
