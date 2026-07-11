param(
    [switch]$AllowUnpinned,
    [string]$Python = ".venv311\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$releasePython = & $Python -c "import sys; print(sys.version_info[:2] == (3, 11) and sys.version_info.releaselevel == 'final')"
if ($LASTEXITCODE -ne 0 -or $releasePython.Trim() -ne "True") {
    throw "Windows release builds require a final Python 3.11 runtime."
}

$verifyArgs = @("scripts\verify_resources.py")
if ($AllowUnpinned) { $verifyArgs += "--allow-unpinned" }
& $Python @verifyArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python -m PyInstaller `
    --noconfirm `
    --clean `
    --windowed `
    --onedir `
    --name=CutVideo `
    --icon="$Root\resources\icons\app-icon.ico" `
    --paths=src `
    --additional-hooks-dir="$Root\scripts\pyinstaller-hooks" `
    --collect-all=pypinyin `
    --exclude-module=modelscope `
    --exclude-module=transformers `
    --exclude-module=huggingface_hub `
    --exclude-module=torch._dynamo `
    --exclude-module=torch._inductor `
    --exclude-module=torch.onnx `
    --exclude-module=torchvision `
    --exclude-module=sklearn `
    --exclude-module=umap `
    --exclude-module=pynndescent `
    --exclude-module=matplotlib `
    --exclude-module=cv2 `
    --add-data="$Root\resources;resources" `
    --add-data="$Root\THIRD_PARTY_NOTICES.md;." `
    --add-data="$Root\README.md;." `
    --distpath=dist `
    --workpath=build\pyinstaller-windows `
    --specpath=build `
    src\cutvideo_launcher.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$report = Join-Path $env:TEMP "cutvideo-selftest-$PID.json"
$env:CUTVIDEO_SELFTEST_REPORT = $report
$selfTest = Start-Process `
    -FilePath (Join-Path $Root "dist\CutVideo\CutVideo.exe") `
    -ArgumentList "--self-test-models" `
    -WindowStyle Hidden `
    -Wait `
    -PassThru
if ($selfTest.ExitCode -ne 0) {
    if (Test-Path -LiteralPath $report) { Get-Content -LiteralPath $report -Encoding UTF8 }
    throw "Frozen application self-test failed with exit code $($selfTest.ExitCode)"
}
Get-Content -LiteralPath $report -Encoding UTF8
Remove-Item -LiteralPath $report -Force
Write-Host "Standalone application created in dist\CutVideo. Distribute the complete directory."
