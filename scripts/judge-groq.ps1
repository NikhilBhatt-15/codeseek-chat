# Usage: powershell -ExecutionPolicy Bypass -File scripts/judge-groq.ps1
# Groq's free tier allows ~65 grades a day for gpt-oss-120b; rerun to continue (grades are cached).
# Grades both systems' saved answers with gpt-oss-120b on Groq (OpenAI-compatible API).
# Asks for the Groq key in a hidden prompt; it is kept only in this process's environment.
$secure = Read-Host -AsSecureString "Paste your Groq API key and press Enter (input is hidden)"
$bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
try {
  $env:JUDGE_API_KEY = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr)
} finally {
  [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
}
if (-not $env:JUDGE_API_KEY) { Write-Host "No key entered; stopping."; exit 1 }
$env:JUDGE_BASE_URL = 'https://api.groq.com/openai/v1'
$env:JUDGE_MODEL = 'openai/gpt-oss-120b'

Set-Location (Join-Path $PSScriptRoot '..')
Write-Host "gpt-oss-120b is grading 128 answers (64 offline + 64 Gemini). Rate limits may make it pause; expect 15-60 minutes."
try {
  npm run eval:answers
} finally {
  Remove-Item Env:JUDGE_API_KEY -ErrorAction SilentlyContinue
  Write-Host "Done. The key has been cleared from this process."
}
