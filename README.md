# Rockoon drift simulator (CMEMS → OpenDrift)

気球が海上に落下した後の漂流（回収船/回収車の判断材料）を **CMEMS → OpenDrift(OceanDrift)** で計算するツールの雛形です。

## ねらい（去年コードからの改善点）
- 1ファイル肥大化をやめて、**CMEMS取得/NetCDF検証/OpenDrift実行/出力** を分離
- `copernicusmarine subset` の結果が壊れてたり時間範囲が足りない時に、**自動でリトライ**して「取れたデータだけで確実に走る」
- 出力に `run_manifest.json` を追加して、**使ったデータセット・時刻範囲・パラメータが必ず追跡できる**

## セットアップ
推奨: Ubuntu (WSL2) + venv

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
```

依存（最低限）:

```bash
pip install opendrift==1.14.3 xarray netcdf4 pandas numpy copernicus-marine-client
```

PNGを出したい場合（重い）:

```bash
pip install matplotlib cartopy
```

## CMEMSログイン（初回のみ）
```bash
copernicusmarine login
copernicusmarine whoami
```

## 実行例
```bash
python cli.py --lon 135.0 --lat 33.5 --time-utc 2025-09-07T10:30:00 --hours 12 --pretty-png
```

風や波が取れない場合:
- `--no-waves`（波/ストークス漂流なし）
- `--no-wind`（風なし、定数風にフォールバック）
