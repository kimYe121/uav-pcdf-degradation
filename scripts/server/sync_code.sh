#!/usr/bin/env bash
# 服务器端代码落地：解压上传的代码包，覆盖代码目录并清理陈旧文件。
#
# 与 rsync --delete 的区别：
#   * 只碰代码目录（src/configs/scripts/tests/docs + 顶层 py/md/toml），
#     绝不触碰 data/ weights/ runs/ logs/；
#   * 覆盖前先打一份代码备份包，出问题可直接回滚；
#   * 先解到临时目录再拷，解压失败不会留下半个仓库。
#
# 用法：bash scripts/server/sync_code.sh [/tmp/pcdf_code.tar.gz] [仓库根]
#       PRUNE=0 只覆盖不删除陈旧文件
set -euo pipefail

TARBALL=${1:-/tmp/pcdf_code.tar.gz}
REPO=${2:-/root/rivermind-data/pcdf}
CODE_DIRS=(src configs scripts tests docs)
TS=$(date +%Y%m%d_%H%M%S)

[ -f "$TARBALL" ] || { echo "[sync] missing code package: $TARBALL" >&2; exit 1; }
[ -d "$REPO" ]    || { echo "[sync] missing repo: $REPO" >&2; exit 1; }

STAGE=$(mktemp -d /tmp/pcdf_stage.XXXXXX)
trap 'rm -rf "$STAGE"' EXIT
echo "[sync] extract $TARBALL -> $STAGE"
tar -xzf "$TARBALL" -C "$STAGE"

echo "[sync] backup current code (data/weights/runs/logs untouched)"
mkdir -p "$REPO/.backup"
BACKUP="$REPO/.backup/code_${TS}.tar.gz"
existing=()
for d in "${CODE_DIRS[@]}"; do [ -d "$REPO/$d" ] && existing+=("$d"); done
for f in pyproject.toml README.md REPRODUCTION.md; do [ -f "$REPO/$f" ] && existing+=("$f"); done
if [ ${#existing[@]} -gt 0 ]; then
  tar -czf "$BACKUP" -C "$REPO" "${existing[@]}"
  echo "[sync] backup: $BACKUP"
fi

# 删除服务器上已有、但新包里没有的文件（陈旧文件），再整目录覆盖。
for d in "${CODE_DIRS[@]}"; do
  [ -d "$STAGE/$d" ] || continue
  if [ -d "$REPO/$d" ] && [ "${PRUNE:-1}" = "1" ]; then
    while IFS= read -r rel; do
      [ -e "$STAGE/$d/$rel" ] || { rm -f "$REPO/$d/$rel"; echo "[sync] prune $d/$rel"; }
    done < <(cd "$REPO/$d" && find . -type f -not -path '*/__pycache__/*' -not -name '*.pyc' | sed 's|^\./||')
    find "$REPO/$d" -type d -empty -delete 2>/dev/null || true
  fi
  mkdir -p "$REPO/$d"
  cp -a "$STAGE/$d/." "$REPO/$d/"
done

for f in pyproject.toml README.md REPRODUCTION.md .gitattributes .gitignore LICENSE; do
  if [ -f "$STAGE/$f" ]; then cp -a "$STAGE/$f" "$REPO/$f"; fi
done

# .pyc 会让"文件换了、字节码还是旧的"这种问题变得难查，统一清掉。
find "$REPO/src" "$REPO/tests" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true

echo "[sync] counts"
for d in "${CODE_DIRS[@]}"; do
  n_stage=$(find "$STAGE/$d" -type f 2>/dev/null | wc -l)
  n_repo=$(find "$REPO/$d" -type f -not -path '*/__pycache__/*' 2>/dev/null | wc -l)
  printf '  %-9s stage=%-4s repo=%s\n' "$d" "$n_stage" "$n_repo"
done
echo "[sync] done"