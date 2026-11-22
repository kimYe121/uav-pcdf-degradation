"""带断点续传的下载工具（CLIP 与骨架权重共用）。

原先这段逻辑只存在于 ``fetch_clip.py`` 里，新增"下载骨架权重"就得复制一份。
现在抽出来：进度打印、``.part`` 续传、体积校验、重试都只有一处实现。
"""

from __future__ import annotations

import time
import urllib.error
import urllib.request
from pathlib import Path

CHUNK = 4 * 1024 * 1024
USER_AGENT = "pcdf-repro/2.0"
DEFAULT_TIMEOUT = 30.0
DEFAULT_RETRIES = 3


def fmt_size(num: float) -> str:
    """人类可读体积。"""
    if num >= 1e9:
        return f"{num / 1e9:.2f} GB"
    if num >= 1e6:
        return f"{num / 1e6:.1f} MB"
    return f"{num / 1e3:.0f} KB"


def download_file(
    url: str,
    dest: Path,
    *,
    min_size: int,
    timeout: float = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
    quiet: bool = False,
) -> None:
    """下载单个文件到 ``dest``；先写 ``.part``，校验体积后再改名。

    Args:
        url: 下载地址。
        dest: 目标文件路径。
        min_size: 最小合理体积，用来识别 HTML 错误页/被截断的下载。
        timeout: 单次连接/读取超时（秒）。
        retries: 失败重试次数（含首次）。
        quiet: 不打印进度（脚本里批量下载时用）。

    Raises:
        OSError: 所有重试都失败，或落盘体积小于 ``min_size``。
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        done = part.stat().st_size if part.exists() else 0
        headers = {"User-Agent": USER_AGENT}
        if done:
            headers["Range"] = f"bytes={done}-"
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                total = int(response.headers.get("Content-Length") or 0)
                if response.status == 206:
                    mode = "ab"
                    total += done
                else:
                    mode, done = "wb", 0
                start = time.monotonic()
                last_print = 0.0
                with open(part, mode) as handle:
                    while True:
                        chunk = response.read(CHUNK)
                        if not chunk:
                            break
                        handle.write(chunk)
                        done += len(chunk)
                        now = time.monotonic()
                        # 最多每 2 秒刷一次，避免日志被进度条刷爆。
                        if not quiet and now - last_print > 2:
                            last_print = now
                            speed = done / max(now - start, 1e-6)
                            if total:
                                pct = min(100.0, done * 100.0 / total)
                                print(
                                    f"    {pct:5.1f}%  {fmt_size(done)} / {fmt_size(total)}"
                                    f"  {fmt_size(speed)}/s",
                                    flush=True,
                                )
                            else:
                                print(f"    {fmt_size(done)}  {fmt_size(speed)}/s", flush=True)
                size = part.stat().st_size
                if size < min_size:
                    raise OSError(f"文件只有 {size} 字节，小于预期 {min_size}")
                part.replace(dest)
                return
        except urllib.error.HTTPError as exc:
            if exc.code == 416 and part.exists():
                part.unlink()  # 续传区间非法，从头再来
            last_error = exc
        except Exception as exc:  # noqa: BLE001 —— 网络类异常统一重试
            last_error = exc
        if attempt < retries:
            print(
                f"    第 {attempt} 次失败（{type(last_error).__name__}: {last_error}），重试…",
                flush=True,
            )
            time.sleep(2)
    raise OSError(f"下载失败：{url}（{last_error}）")


def download_first(
    urls: list[str],
    dest: Path,
    *,
    min_size: int,
    timeout: float = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
) -> str:
    """按顺序尝试多个镜像，返回实际成功的 URL。"""
    errors: list[str] = []
    for url in urls:
        print(f"  [下载] {url}", flush=True)
        try:
            download_file(url, dest, min_size=min_size, timeout=timeout, retries=retries)
            return url
        except Exception as exc:  # noqa: BLE001 —— 换下一个镜像
            print(f"    [warn] 失败：{type(exc).__name__}: {exc}", flush=True)
            errors.append(f"{url}: {exc}")
    joined = "\n".join(f"  - {line}" for line in errors)
    raise OSError(f"所有镜像都失败：\n{joined}")
