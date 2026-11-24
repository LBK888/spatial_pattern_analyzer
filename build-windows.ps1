param(
    [string]$Python = "python",
    [string]$Output = "dist"
)
$ErrorActionPreference = "Stop"
Push-Location -LiteralPath $PSScriptRoot
$origPath = $env:PATH
try {
    $pythonExe = (Get-Command $Python -CommandType Application | Select-Object -First 1).Source
    # PyInstaller searches PATH for DLLs. Third-party ICU and UCRT from other
    # applications can shadow the Windows versions that Qt imports.
    $env:PATH = @((Split-Path -Parent $pythonExe), (Join-Path $env:SystemRoot "System32"), $env:SystemRoot) -join ';'
    & $pythonExe -B -m unittest discover -s tests
    if ($LASTEXITCODE -ne 0) { throw "Tests failed." }
    $buildArgs = @(
        "--noconfirm", "--clean", "--onefile", "--windowed", "--disable-windowed-traceback",
        "--name", "SpatialPatternAnalyzer-Windows-x64",
        "--distpath", $Output, "--workpath", ".build/pyinstaller", "--specpath", ".build",
        "--hidden-import", "xlrd",
        "--hidden-import", "PyQt6.sip",
        "--hidden-import", "PyQt6.QtSvg",
        "--hidden-import", "matplotlib.backends.backend_pdf",
        "--hidden-import", "matplotlib.backends.backend_svg",
        "--hidden-import", "matplotlib.backends.backend_qtagg",
        "--exclude-module", "tkinter",
        "--exclude-module", "PyQt5",
        "--exclude-module", "PySide6",
        "--add-data", "$PSScriptRoot/demo/points.csv:demo", "--add-data", "$PSScriptRoot/README.md:.",
        "--add-data", "$PSScriptRoot/README.zh-TW.md:.", "--add-data", "$PSScriptRoot/CITATION.bib:.",
        "--add-data", "$PSScriptRoot/LICENSE:.",
        "--add-data", "$PSScriptRoot/ui/checked.svg:ui",
        "$PSScriptRoot/spatial_analysis_gui.py"
    )
    & $pythonExe -m PyInstaller @buildArgs
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller build failed." }
    $exe = Join-Path $Output "SpatialPatternAnalyzer-Windows-x64.exe"
    $check = Join-Path $PSScriptRoot (".build/smoke-" + [guid]::NewGuid().ToString("N"))
    $processArgs = @("--smoke-test", ('"' + $check + '"'))
    $process = Start-Process -FilePath (Resolve-Path $exe) -ArgumentList $processArgs -WindowStyle Hidden -PassThru
    if (-not $process.WaitForExit(50000)) {
        Stop-Process -Id $process.Id -ErrorAction SilentlyContinue
        throw "Packaged application smoke test timed out."
    }
    if ($process.ExitCode -ne 0) {
        $diagnostic = Join-Path $check "error.txt"
        if (Test-Path $diagnostic) { Get-Content -LiteralPath $diagnostic }
        throw "Packaged application smoke test failed."
    }
    foreach ($file in @("groups-en.pdf", "groups-zh-TW.pdf", "results.xlsx")) {
        if (-not (Test-Path (Join-Path $check $file))) { throw "Smoke test did not produce $file." }
    }
    (Get-FileHash $exe -Algorithm SHA256).Hash.ToLowerInvariant() + "  " + (Split-Path $exe -Leaf) |
        Set-Content -Encoding ascii (Join-Path $Output "SHA256.txt")
} finally {
    $env:PATH = $origPath
    Pop-Location
}
