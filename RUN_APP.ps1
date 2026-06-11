$ErrorActionPreference = "Stop"
$Project = Join-Path $env:USERPROFILE "Downloads\joseph_carlson_finfluencer_pipeline"
if (!(Test-Path $Project)) {
    Write-Host "Project folder not found: $Project"
    exit 1
}
Set-Location $Project
python -m pip install -r .\requirements.txt
python -m streamlit run .\streamlit_app.py
