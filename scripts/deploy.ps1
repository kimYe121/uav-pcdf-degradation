# 本地 -> 服务器代码同步：tar 打包 + scp 上传 + 远端解压覆盖
#
# 为什么不用 rsync：
#   * Windows 上 rsync 需要额外装 cwRsync/MSYS，路径与权限还容易踩坑；
#   * 代码与数据在服务器上是同一个目录，rsync --delete 一旦 exclude 写漏
#     就会删到 data/weights，风险不对称。
#   tar.gz 只有二十几兆，scp 一下就完事；服务端脚本先备份再覆盖，可回滚。
#
# 用法：
#   pwsh scripts/deploy.ps1                 # 打包 + 上传 + 远端覆盖
#   pwsh scripts/deploy.ps1 -DryRun         # 只打包，看看会带哪些文件
#   pwsh scripts/deploy.ps1 -NoPrune        # 覆盖但不清理陈旧文件
param(
    [string]$SshHost   = "user@your-host",
    [int]   $Port      = 22,
    [string]$RemoteDir = "/root/rivermind-data/pcdf",
    [switch]$DryRun,
    [switch]$NoPrune
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Push-Location $root
try {
    $stamp   = Get-Date -Format "yyyyMMdd_HHmmss"
    $archive = Join-Path $env:TEMP "pcdf_code_$stamp.tar.gz"

    $excludes = @(
        "--exclude=./.git", "--exclude=./data", "--exclude=./weights",
        "--exclude=./runs", "--exclude=./logs", "--exclude=./archive",
        "--exclude=./output", "--exclude=./.workbuddy", "--exclude=./.vscode",
        "--exclude=./.pytest_cache", "--exclude=__pycache__", "--exclude=*.pyc",
        "--exclude=*.cache", "--exclude=./.backup", "--exclude=./debug.log"
    )

    Write-Host "[deploy] 打包 -> $archive"
    & tar -czf $archive @excludes -C $root .
    if ($LASTEXITCODE -ne 0) { throw "tar 打包失败 ($LASTEXITCODE)" }
    $size = [math]::Round((Get-Item $archive).Length / 1MB, 2)
    $files = (& tar -tzf $archive | Measure-Object -Line).Lines
    Write-Host "[deploy] 包大小 ${size} MB，$files 个条目"
    if ($DryRun) {
        & tar -tzf $archive | Select-Object -First 40
        Write-Host "[deploy] -DryRun：不上传"
        return
    }

    $remotePkg = "/tmp/pcdf_code.tar.gz"
    Write-Host "[deploy] 上传 -> ${SshHost}:$remotePkg (port $Port)"
    & scp -P $Port -o BatchMode=yes $archive "${SshHost}:$remotePkg"
    if ($LASTEXITCODE -ne 0) { throw "scp 上传失败 ($LASTEXITCODE)" }

    $prune = if ($NoPrune) { "0" } else { "1" }
    # 先单独上传同步脚本：第一次部署时仓库里可能还没有它，不能依赖包内路径。
    $remoteSync = "/tmp/pcdf_sync_code.sh"
    & scp -P $Port -o BatchMode=yes (Join-Path $root "scripts/server/sync_code.sh") "${SshHost}:$remoteSync"
    if ($LASTEXITCODE -ne 0) { throw "scp 同步脚本失败 ($LASTEXITCODE)" }

    Write-Host "[deploy] 远端解压覆盖 -> $RemoteDir"
    & ssh -p $Port -o BatchMode=yes $SshHost "PRUNE=$prune bash $remoteSync $remotePkg $RemoteDir"
    if ($LASTEXITCODE -ne 0) { throw "远端同步失败 ($LASTEXITCODE)" }

    Write-Host "[deploy] 校验：远端文件数与关键文件时间戳"
    & ssh -p $Port -o BatchMode=yes $SshHost "cd $RemoteDir && find src configs scripts tests docs -type f | wc -l && ls -l src/pcdf/models/encoder.py tests/test_baseline.py"
}
finally {
    Pop-Location
}