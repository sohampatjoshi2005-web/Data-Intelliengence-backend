from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


class LocalFeatureStore:
    """Minimal online/offline feature store facade.

    Offline: parquet files by entity/table.
    Online: latest features per key from parquet snapshot.
    """

    def __init__(self) -> None:
        self.root = Path(os.getenv("FEATURE_STORE_DIR", "artifacts/feature_store"))
        self.offline_dir = self.root / "offline"
        self.online_dir = self.root / "online"
        self.offline_dir.mkdir(parents=True, exist_ok=True)
        self.online_dir.mkdir(parents=True, exist_ok=True)
        self.file_ext = ".parquet" if self._parquet_available() else ".pkl"

    def _parquet_available(self) -> bool:
        return importlib.util.find_spec("pyarrow") is not None or importlib.util.find_spec("fastparquet") is not None

    def _table_path(self, directory: Path, table: str, suffix: str) -> Path:
        safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in table)
        return directory / f"{safe}{suffix}"

    def _resolve_existing_path(self, directory: Path, table: str) -> Path:
        for suffix in (".parquet", ".pkl"):
            path = self._table_path(directory, table, suffix)
            if path.exists():
                return path
        return self._table_path(directory, table, self.file_ext)

    def _read_frame(self, path: Path) -> pd.DataFrame:
        if path.suffix == ".pkl":
            return pd.read_pickle(path)
        return pd.read_parquet(path)

    def _write_frame(self, df: pd.DataFrame, path: Path) -> Path:
        if path.suffix == ".pkl":
            df.to_pickle(path)
            return path

        try:
            df.to_parquet(path, index=False)
            return path
        except ImportError:
            fallback = path.with_suffix(".pkl")
            df.to_pickle(fallback)
            return fallback

    def _offline_path(self, table: str) -> Path:
        return self._resolve_existing_path(self.offline_dir, table)

    def _online_path(self, table: str) -> Path:
        return self._resolve_existing_path(self.online_dir, table)

    def upsert_offline(self, table: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        df_new = pd.DataFrame(rows)
        path = self._offline_path(table)
        if path.exists():
            try:
                old = self._read_frame(path)
                df = pd.concat([old, df_new], ignore_index=True)
            except Exception:
                df = df_new.copy()
        else:
            df = df_new.copy()
        written_path = self._write_frame(df, path)
        return {"ok": True, "table": table, "rows": int(len(df_new)), "offline_path": str(written_path)}

    def materialize_online(self, table: str, key_col: str, ts_col: str | None = None) -> Dict[str, Any]:
        path = self._offline_path(table)
        if not path.exists():
            return {"ok": False, "error": "offline_table_missing"}
        df = self._read_frame(path)
        if key_col not in df.columns:
            return {"ok": False, "error": f"missing_key_col:{key_col}"}

        if ts_col and ts_col in df.columns:
            df = df.sort_values(ts_col)
        latest = df.drop_duplicates(subset=[key_col], keep="last")

        opath = self._online_path(table)
        written_path = self._write_frame(latest, opath)
        return {"ok": True, "table": table, "rows": int(len(latest)), "online_path": str(written_path)}

    def read_online(self, table: str, key_col: str, key_val: Any) -> Dict[str, Any]:
        path = self._online_path(table)
        if not path.exists():
            return {"ok": False, "error": "online_table_missing"}
        df = self._read_frame(path)
        if key_col not in df.columns:
            return {"ok": False, "error": f"missing_key_col:{key_col}"}
        out = df[df[key_col].astype(str) == str(key_val)]
        return {"ok": True, "table": table, "record": out.head(1).to_dict(orient="records")[0] if not out.empty else None}

    def list_tables(self) -> Dict[str, Any]:
        offline = sorted({p.stem for suffix in ("*.parquet", "*.pkl") for p in self.offline_dir.glob(suffix)})
        online = sorted({p.stem for suffix in ("*.parquet", "*.pkl") for p in self.online_dir.glob(suffix)})
        return {"offline_tables": offline, "online_tables": online}
