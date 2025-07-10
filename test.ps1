$Distro = "audio-lora"

# 1. Удаляем кэш whisper из WSL
Write-Host "`n🧹 Удаляем кэш модели в WSL..."
wsl -d $Distro -- bash -c "rm -rf ~/.cache/huggingface"

# 2. Копируем из Windows в WSL
$Src = "/mnt/d/VM/WSL2/audio-lora-builder/temp/huggingface/whisper/hub/"
$Dst = "~/.cache/huggingface/hub/"
Write-Host "📦 Копируем модель из Windows в WSL..."
wsl -d $Distro -- bash -c "mkdir -p $Dst && cp -r '${Src}'* $Dst"

# 3. Проверка симлинков в кэше WSL
Write-Host "`n🔎 Проверяем симлинки в WSL..."
$CheckCmd = "find ~/.cache/huggingface/hub -type l ! -exec test -e {} \; -print"

$Broken = wsl -d $Distro -- bash -c "$CheckCmd"

if ($Broken) {
    Write-Host "❌ Обнаружены битые симлинки:"
    Write-Host $Broken
    Write-Host "`n😢 Симлинк-кэш модели повреждён"
} else {
    Write-Host "✅ Все симлинки модели целы. Кэш готов к использованию 🎉"
}




