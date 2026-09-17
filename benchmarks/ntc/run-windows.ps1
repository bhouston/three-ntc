param(
  [string]$Label = 'rtx3060ti-4000',
  [switch]$SmokeOnly,
  [ValidateSet('chromium','chrome','msedge')][string]$BrowserChannel = 'chromium',
  [switch]$Headed
)
$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '../..')
if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
  throw 'Install Node.js 24 from https://nodejs.org/ then reopen PowerShell.'
}
$major = [int]((node -p 'parseInt(process.versions.node)').Trim())
if ($major -lt 24) { throw 'This launcher requires Node.js 24 or newer.' }
function Run-Checked {
  param([string]$Program, [string[]]$Arguments)
  & $Program @Arguments
  if ($LASTEXITCODE -ne 0) { throw "$Program exited with code $LASTEXITCODE" }
}
# Explicit native Windows WebGPU backend. Do not inherit software-CI flags.
$env:CI = $null
$env:NTC_BROWSER = $null
$env:NTC_GPU_BACKEND = 'native'
Run-Checked 'npx.cmd' @('--yes','pnpm@11.1.3','install','--frozen-lockfile')
Run-Checked 'npx.cmd' @('--yes','pnpm@11.1.3','exec','playwright','install','chromium')

$diagnosticArgs = @('benchmarks/ntc/diagnose-gpu.mjs','--channel',$BrowserChannel,'--expected-vendor','nvidia')
if ($Headed) { $diagnosticArgs += '--headed' }
Run-Checked 'node' $diagnosticArgs
$diagnostic = Get-Content 'output/benchmarks/gpu-diagnostics.json' -Raw | ConvertFrom-Json
$env:NTC_BROWSER_CHANNEL = $diagnostic.selected.channel
$env:NTC_GPU_HEADED = if ($diagnostic.selected.headed) { '1' } else { '0' }


$plan = Join-Path "output/benchmarks/$Label" 'plan.json'
if ($SmokeOnly -or -not (Test-Path $plan)) {
  $smoke = "$Label-smoke-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
  Run-Checked 'node' @('benchmarks/ntc/run.mjs','--label',$smoke,'--materials','uv_grid','--variants','reference','--seeds','1','--iterations','8','--resolution','64')
  $result = Get-Content "output/benchmarks/$smoke/results/uv_grid__reference__seed-1.json" -Raw | ConvertFrom-Json
  if ($result.status -ne 'complete') { throw "GPU smoke check failed. See output/benchmarks/$smoke/logs." }
  $adapter = $result.adapter | ConvertTo-Json -Compress
  Write-Host "WebGPU adapter: $adapter"
  if ($adapter -notmatch '(?i)nvidia|10de') {
    throw 'The smoke test did not identify NVIDIA. Select the NVIDIA GPU in Windows Graphics settings for the Playwright Chromium executable and retry.'
  }
  if ($result.gradientPrecision -ne 'float') { throw 'Native paper batches require floating accumulation; this GPU/compiler selected the fixed-point fallback.' }
  if ($SmokeOnly) { Write-Host 'NVIDIA smoke check passed.'; exit 0 }
}
Write-Host "Starting/resuming $Label. Keep this terminal open and prevent the PC from sleeping."
Run-Checked 'node' @('benchmarks/ntc/run.mjs','--label',$Label)
Write-Host "Report: output/benchmarks/$Label/report.html"
