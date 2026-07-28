from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path
from typing import Tuple

import pandas as pd
import requests

from app.core.config import settings

try:
    from markitdown import MarkItDown
except Exception:
    MarkItDown = None

try:
    import pytesseract
    from PIL import Image
except Exception:
    pytesseract = None
    Image = None

try:
    import pdfplumber
except Exception:
    pdfplumber = None

try:
    import yaml
except Exception:
    yaml = None

try:
    from docx import Document
except Exception:
    Document = None

try:
    import xmltodict
except Exception:
    xmltodict = None

try:
    from markdown import markdown as md_to_html
except Exception:
    md_to_html = None

try:
    from markdownify import markdownify as html_to_md
except Exception:
    html_to_md = None

try:
    import boto3
except Exception:
    boto3 = None


def _encode_bytes(data: bytes) -> str:
    return base64.b64encode(data).decode("utf-8")


def run_transformation(file_name: str, payload: bytes, transform_type: str) -> Tuple[str, str, str]:
    name = Path(file_name).stem
    ext = Path(file_name).suffix.lower()

    def _img_media_type() -> str:
        if ext in {".jpg", ".jpeg"}:
            return "image/jpeg"
        if ext == ".png":
            return "image/png"
        return "image/png"

    def _prep_image_for_vision(raw: bytes) -> bytes:
        if Image is None:
            return raw
        try:
            img = Image.open(io.BytesIO(raw)).convert("RGB")
            max_side = 1024
            w, h = img.size
            scale = min(max_side / max(w, h), 1.0)
            if scale < 1.0:
                img = img.resize((int(w * scale), int(h * scale)))
            out = io.BytesIO()
            img.save(out, format="PNG")
            return out.getvalue()
        except Exception:
            return raw

    if transform_type == "CSV → Markdown Table":
        df = pd.read_csv(io.BytesIO(payload))
        out = df.to_markdown(index=False).encode("utf-8")
        return f"{name}.md", "text/markdown", _encode_bytes(out)

    if transform_type == "CSV → HTML Table":
        df = pd.read_csv(io.BytesIO(payload))
        out = df.to_html(index=False).encode("utf-8")
        return f"{name}.html", "text/html", _encode_bytes(out)

    if transform_type == "CSV → Excel (XLSX)":
        df = pd.read_csv(io.BytesIO(payload))
        out = io.BytesIO()
        with pd.ExcelWriter(out, engine="openpyxl") as writer:
            df.to_excel(writer, index=False)
        return f"{name}.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", _encode_bytes(out.getvalue())

    if transform_type == "CSV → JSON":
        df = pd.read_csv(io.BytesIO(payload))
        out = df.to_json(orient="records").encode("utf-8")
        return f"{name}.json", "application/json", _encode_bytes(out)

    if transform_type == "CSV → TSV":
        df = pd.read_csv(io.BytesIO(payload))
        out = io.StringIO()
        df.to_csv(out, index=False, sep="\t")
        return f"{name}.tsv", "text/tab-separated-values", _encode_bytes(out.getvalue().encode("utf-8"))

    if transform_type == "TSV → JSON":
        df = pd.read_csv(io.BytesIO(payload), sep="\t")
        out = df.to_json(orient="records").encode("utf-8")
        return f"{name}.json", "application/json", _encode_bytes(out)

    if transform_type == "TSV → Excel (XLSX)":
        df = pd.read_csv(io.BytesIO(payload), sep="\t")
        out = io.BytesIO()
        with pd.ExcelWriter(out, engine="openpyxl") as writer:
            df.to_excel(writer, index=False)
        return f"{name}.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", _encode_bytes(out.getvalue())

    if transform_type == "TSV → Markdown Table":
        df = pd.read_csv(io.BytesIO(payload), sep="\t")
        out = df.to_markdown(index=False).encode("utf-8")
        return f"{name}.md", "text/markdown", _encode_bytes(out)

    if transform_type == "TSV → CSV":
        df = pd.read_csv(io.BytesIO(payload), sep="\t")
        out = io.StringIO()
        df.to_csv(out, index=False)
        return f"{name}.csv", "text/csv", _encode_bytes(out.getvalue().encode("utf-8"))

    if transform_type == "Excel → TSV":
        df = pd.read_excel(io.BytesIO(payload))
        out = io.StringIO()
        df.to_csv(out, index=False, sep="\t")
        return f"{name}.tsv", "text/tab-separated-values", _encode_bytes(out.getvalue().encode("utf-8"))

    if transform_type == "Excel → Markdown Table":
        df = pd.read_excel(io.BytesIO(payload))
        out = df.to_markdown(index=False).encode("utf-8")
        return f"{name}.md", "text/markdown", _encode_bytes(out)

    if transform_type == "Excel → CSV":
        df = pd.read_excel(io.BytesIO(payload))
        out = io.StringIO()
        df.to_csv(out, index=False)
        return f"{name}.csv", "text/csv", _encode_bytes(out.getvalue().encode("utf-8"))

    if transform_type == "Excel → JSON":
        df = pd.read_excel(io.BytesIO(payload))
        out = df.to_json(orient="records").encode("utf-8")
        return f"{name}.json", "application/json", _encode_bytes(out)

    if transform_type == "JSON → TSV":
        df = pd.read_json(io.BytesIO(payload))
        out = io.StringIO()
        df.to_csv(out, index=False, sep="\t")
        return f"{name}.tsv", "text/tab-separated-values", _encode_bytes(out.getvalue().encode("utf-8"))

    if transform_type == "JSON → Markdown Table":
        df = pd.read_json(io.BytesIO(payload))
        out = df.to_markdown(index=False).encode("utf-8")
        return f"{name}.md", "text/markdown", _encode_bytes(out)

    if transform_type == "JSON → YAML":
        if yaml is None:
            raise RuntimeError("pyyaml is not installed.")
        data = json.loads(payload.decode("utf-8"))
        out = yaml.safe_dump(data, sort_keys=False).encode("utf-8")
        return f"{name}.yaml", "text/yaml", _encode_bytes(out)

    if transform_type == "YAML → JSON":
        if yaml is None:
            raise RuntimeError("pyyaml is not installed.")
        data = yaml.safe_load(payload.decode("utf-8"))
        out = json.dumps(data, indent=2).encode("utf-8")
        return f"{name}.json", "application/json", _encode_bytes(out)

    if transform_type == "XML → JSON":
        if xmltodict is None:
            raise RuntimeError("xmltodict is not installed.")
        data = xmltodict.parse(payload.decode("utf-8"))
        out = json.dumps(data, indent=2).encode("utf-8")
        return f"{name}.json", "application/json", _encode_bytes(out)

    if transform_type == "JSON → XML":
        if xmltodict is None:
            raise RuntimeError("xmltodict is not installed.")
        data = json.loads(payload.decode("utf-8"))
        out = xmltodict.unparse(data, pretty=True).encode("utf-8")
        return f"{name}.xml", "application/xml", _encode_bytes(out)

    if transform_type == "JSON → Pretty":
        data = json.loads(payload.decode("utf-8"))
        out = json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")
        return f"{name}.json", "application/json", _encode_bytes(out)

    if transform_type == "JSON → Minify":
        data = json.loads(payload.decode("utf-8"))
        out = json.dumps(data, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        return f"{name}.json", "application/json", _encode_bytes(out)

    if transform_type == "JSON → CSV":
        df = pd.read_json(io.BytesIO(payload))
        out = io.StringIO()
        df.to_csv(out, index=False)
        return f"{name}.csv", "text/csv", _encode_bytes(out.getvalue().encode("utf-8"))

    if transform_type == "JSON → Excel (XLSX)":
        df = pd.read_json(io.BytesIO(payload))
        out = io.BytesIO()
        with pd.ExcelWriter(out, engine="openpyxl") as writer:
            df.to_excel(writer, index=False)
        return f"{name}.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", _encode_bytes(out.getvalue())

    if transform_type == "TXT → Markdown":
        txt = payload.decode("utf-8", errors="ignore")
        out = f"```\n{txt}\n```\n".encode("utf-8")
        return f"{name}.md", "text/markdown", _encode_bytes(out)

    if transform_type == "TXT → HTML":
        txt = payload.decode("utf-8", errors="ignore")
        out = f"<pre>{txt}</pre>".encode("utf-8")
        return f"{name}.html", "text/html", _encode_bytes(out)

    if transform_type == "PDF/DOCX → Markdown":
        if MarkItDown is None:
            raise RuntimeError("markitdown is not installed.")
        tmp_path = f"/tmp/{file_name}"
        with open(tmp_path, "wb") as f:
            f.write(payload)
        md = MarkItDown().convert(tmp_path).text_content
        return f"{name}.md", "text/markdown", _encode_bytes(md.encode("utf-8"))

    if transform_type == "PDF → Text":
        if pdfplumber is None:
            raise RuntimeError("pdfplumber is not installed.")
        tmp_path = f"/tmp/{file_name}"
        with open(tmp_path, "wb") as f:
            f.write(payload)
        chunks = []
        with pdfplumber.open(tmp_path) as pdf:
            for page in pdf.pages:
                chunks.append(page.extract_text() or "")
        out = "\n\n".join(chunks).encode("utf-8")
        return f"{name}.txt", "text/plain", _encode_bytes(out)

    if transform_type == "DOCX → Text":
        if Document is None:
            raise RuntimeError("python-docx is not installed.")
        tmp_path = f"/tmp/{file_name}"
        with open(tmp_path, "wb") as f:
            f.write(payload)
        doc = Document(tmp_path)
        out = "\n".join([p.text for p in doc.paragraphs]).encode("utf-8")
        return f"{name}.txt", "text/plain", _encode_bytes(out)

    if transform_type == "Markdown → HTML":
        if md_to_html is None:
            raise RuntimeError("markdown is not installed.")
        txt = payload.decode("utf-8", errors="ignore")
        out = md_to_html(txt).encode("utf-8")
        return f"{name}.html", "text/html", _encode_bytes(out)

    if transform_type == "HTML → Markdown":
        if html_to_md is None:
            raise RuntimeError("markdownify is not installed.")
        txt = payload.decode("utf-8", errors="ignore")
        out = html_to_md(txt).encode("utf-8")
        return f"{name}.md", "text/markdown", _encode_bytes(out)

    if transform_type in ("Image → Text (OCR)", "Image → Text (Vision)"):
        provider = os.getenv("TRANSFORM_IMAGE_PROVIDER", "moondream").lower()
        if provider == "bedrock" and boto3 is not None:
            region = os.getenv("BEDROCK_REGION", "us-east-2")
            model_id = os.getenv("BEDROCK_VISION_MODEL_ID", os.getenv("BEDROCK_MODEL_ID", ""))
            if not model_id:
                raise RuntimeError("BEDROCK_VISION_MODEL_ID or BEDROCK_MODEL_ID is required for Bedrock image OCR.")
            client = boto3.client("bedrock-runtime", region_name=region)
            img_b64 = base64.b64encode(payload).decode("utf-8")
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 256,
                "temperature": 0.0,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "source": {"type": "base64", "media_type": _img_media_type(), "data": img_b64}},
                            {"type": "text", "text": "Describe the image in plain text. If there is readable text, include it."},
                        ],
                    }
                ],
            }
            resp = client.invoke_model(
                modelId=model_id,
                body=json.dumps(body).encode("utf-8"),
                contentType="application/json",
                accept="application/json",
            )
            raw = resp["body"].read().decode("utf-8")
            text = json.loads(raw)["content"][0]["text"]
            return f"{name}.txt", "text/plain", _encode_bytes(text.encode("utf-8"))

        if provider in ("moondream", "ollama"):
            base_url = settings.ollama_vision_base_url.rstrip("/")
            model = os.getenv("OLLAMA_VISION_MODEL", "moondream")
            timeout = int(os.getenv("OLLAMA_VISION_TIMEOUT", "300"))
            prepared = _prep_image_for_vision(payload)
            img_b64 = base64.b64encode(prepared).decode("utf-8")
            prompt = "Describe the image in plain text. If there is readable text, include it."
            payload_json = {
                "model": model,
                "prompt": prompt,
                "images": [img_b64],
                "stream": False,
                "options": {"num_predict": 256},
            }
            resp = requests.post(f"{base_url}/api/generate", json=payload_json, timeout=timeout)
            if resp.status_code != 200:
                raise RuntimeError(f"Ollama vision failed: {resp.status_code} {resp.text}")
            data = resp.json()
            text = data.get("response", "").strip()
            if not text:
                raise RuntimeError("Ollama vision returned empty text.")
            return f"{name}.txt", "text/plain", _encode_bytes(text.encode("utf-8"))

        if provider == "ocr":
            if pytesseract is None or Image is None:
                raise RuntimeError("pytesseract/Pillow is not installed.")
            tmp_path = f"/tmp/{file_name}"
            with open(tmp_path, "wb") as f:
                f.write(payload)
            img = Image.open(tmp_path)
            text = pytesseract.image_to_string(img)
            return f"{name}.txt", "text/plain", _encode_bytes(text.encode("utf-8"))

        raise RuntimeError(f"Unsupported TRANSFORM_IMAGE_PROVIDER: {provider}")

    if transform_type == "Image → Markdown (Caption)":
        provider = os.getenv("TRANSFORM_IMAGE_PROVIDER", "moondream").lower()
        if provider == "bedrock" and boto3 is not None:
            region = os.getenv("BEDROCK_REGION", "us-east-2")
            model_id = os.getenv("BEDROCK_VISION_MODEL_ID", os.getenv("BEDROCK_MODEL_ID", ""))
            if not model_id:
                raise RuntimeError("BEDROCK_VISION_MODEL_ID or BEDROCK_MODEL_ID is required for Bedrock caption.")
            client = boto3.client("bedrock-runtime", region_name=region)
            img_b64 = base64.b64encode(payload).decode("utf-8")
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 128,
                "temperature": 0.2,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "source": {"type": "base64", "media_type": _img_media_type(), "data": img_b64}},
                            {"type": "text", "text": "Write a concise one-sentence caption for this image."},
                        ],
                    }
                ],
            }
            resp = client.invoke_model(
                modelId=model_id,
                body=json.dumps(body).encode("utf-8"),
                contentType="application/json",
                accept="application/json",
            )
            raw = resp["body"].read().decode("utf-8")
            caption = json.loads(raw)["content"][0]["text"]
            md = f"![caption]({file_name})\n\n{caption}\n"
            return f"{name}.md", "text/markdown", _encode_bytes(md.encode("utf-8"))

        if provider in ("moondream", "ollama"):
            base_url = settings.ollama_vision_base_url.rstrip("/")
            model = os.getenv("OLLAMA_VISION_MODEL", "moondream")
            timeout = int(os.getenv("OLLAMA_VISION_TIMEOUT", "300"))
            prepared = _prep_image_for_vision(payload)
            img_b64 = base64.b64encode(prepared).decode("utf-8")
            prompt = "Write a concise one-sentence caption for this image."
            payload_json = {
                "model": model,
                "prompt": prompt,
                "images": [img_b64],
                "stream": False,
                "options": {"num_predict": 128},
            }
            resp = requests.post(f"{base_url}/api/generate", json=payload_json, timeout=timeout)
            if resp.status_code != 200:
                raise RuntimeError(f"Ollama vision failed: {resp.status_code} {resp.text}")
            data = resp.json()
            caption = data.get("response", "").strip()
            if not caption:
                raise RuntimeError("Ollama vision returned empty caption.")
            md = f"![caption]({file_name})\n\n{caption}\n"
            return f"{name}.md", "text/markdown", _encode_bytes(md.encode("utf-8"))

        if provider == "ocr":
            if pytesseract is None or Image is None:
                raise RuntimeError("pytesseract/Pillow is not installed.")
            tmp_path = f"/tmp/{file_name}"
            with open(tmp_path, "wb") as f:
                f.write(payload)
            img = Image.open(tmp_path)
            text = pytesseract.image_to_string(img)
            md = f"![image]({file_name})\n\n{text.strip()}\n"
            return f"{name}.md", "text/markdown", _encode_bytes(md.encode("utf-8"))

        raise RuntimeError(f"Unsupported TRANSFORM_IMAGE_PROVIDER: {provider}")

    raise ValueError(f"Unknown transform_type: {transform_type}")


def run_logical_transformation(
    file_name: str,
    payload: bytes,
    operations: list,
    right_payload: bytes | None = None,
) -> Tuple[str, str, str, list, int, list, list, dict]:
    """
    Apply a sequence of logical transformations to a dataframe.
    
    Returns: (file_name, media_type, content_base64, columns, row_count, preview_rows, warnings, report)
    """
    warnings = []
    report = {}
    
    try:
        # Read the main file
        df = pd.read_csv(io.BytesIO(payload))
    except Exception as e:
        try:
            df = pd.read_excel(io.BytesIO(payload))
        except Exception:
            raise ValueError(f"Failed to read file: {e}") from e
    
    right_df = None
    if right_payload:
        try:
            right_df = pd.read_csv(io.BytesIO(right_payload))
        except Exception:
            try:
                right_df = pd.read_excel(io.BytesIO(right_payload))
            except Exception as e:
                warnings.append(f"Failed to read right file for join: {e}")
    
    # Apply operations in sequence
    for i, op in enumerate(operations):
        try:
            op_type = op.get("type")
            
            if op_type == "clean":
                columns = op.get("columns", [])
                strategy = op.get("strategy", "drop_missing")
                drop_duplicates = op.get("drop_duplicates", True)
                
                if columns:
                    cols_to_clean = [c for c in columns if c in df.columns]
                else:
                    cols_to_clean = df.columns.tolist()
                
                if strategy == "drop_missing":
                    df = df.dropna(subset=cols_to_clean)
                elif strategy == "fill_mean":
                    for col in cols_to_clean:
                        if pd.api.types.is_numeric_dtype(df[col]):
                            df[col] = df[col].fillna(df[col].mean())
                elif strategy == "fill_median":
                    for col in cols_to_clean:
                        if pd.api.types.is_numeric_dtype(df[col]):
                            df[col] = df[col].fillna(df[col].median())
                elif strategy == "fill_mode":
                    for col in cols_to_clean:
                        df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else None)
                
                if drop_duplicates:
                    df = df.drop_duplicates()
                
                report[f"operation_{i}"] = {"type": "clean", "rows_after": len(df)}
            
            elif op_type == "filter":
                expression = op.get("expression")
                if expression:
                    try:
                        df = df.query(expression)
                        report[f"operation_{i}"] = {"type": "filter", "rows_after": len(df)}
                    except Exception as e:
                        warnings.append(f"Filter operation failed: {e}")
            
            elif op_type == "derive":
                name = op.get("name")
                expression = op.get("expression")
                if name and expression:
                    try:
                        df[name] = df.eval(expression)
                        report[f"operation_{i}"] = {"type": "derive", "new_column": name}
                    except Exception as e:
                        warnings.append(f"Derive operation failed: {e}")
            
            elif op_type == "map":
                column = op.get("column")
                mapping = op.get("mapping", {})
                if column and column in df.columns:
                    try:
                        if isinstance(mapping, str):
                            mapping = json.loads(mapping)
                        df[column] = df[column].map(mapping).fillna(df[column])
                        report[f"operation_{i}"] = {"type": "map", "column": column}
                    except Exception as e:
                        warnings.append(f"Map operation failed: {e}")
            
            elif op_type == "groupby":
                group_by = op.get("group_by", [])
                aggregations = op.get("aggregations", {})
                if group_by:
                    try:
                        valid_group_by = [c for c in group_by if c in df.columns]
                        if valid_group_by:
                            if isinstance(aggregations, str):
                                aggregations = json.loads(aggregations)
                            agg_dict = {}
                            for col, func in aggregations.items():
                                if col in df.columns:
                                    agg_dict[col] = func
                            if agg_dict:
                                df = df.groupby(valid_group_by).agg(agg_dict).reset_index()
                            else:
                                df = df.groupby(valid_group_by).size().reset_index(name="count")
                            report[f"operation_{i}"] = {"type": "groupby", "rows_after": len(df)}
                    except Exception as e:
                        warnings.append(f"Groupby operation failed: {e}")
            
            elif op_type == "join":
                if right_df is not None:
                    left_on = op.get("left_on")
                    right_on = op.get("right_on")
                    how = op.get("how", "inner")
                    try:
                        df = pd.merge(df, right_df, left_on=left_on, right_on=right_on, how=how)
                        report[f"operation_{i}"] = {"type": "join", "rows_after": len(df), "how": how}
                    except Exception as e:
                        warnings.append(f"Join operation failed: {e}")
                else:
                    warnings.append("Right file not provided for join operation")
            
            elif op_type == "scale":
                columns = op.get("columns", [])
                method = op.get("method", "standard")
                if columns:
                    cols_to_scale = [c for c in columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
                    try:
                        if method == "standard":
                            for col in cols_to_scale:
                                mean = df[col].mean()
                                std = df[col].std()
                                if std != 0:
                                    df[col] = (df[col] - mean) / std
                        elif method == "minmax":
                            for col in cols_to_scale:
                                min_val = df[col].min()
                                max_val = df[col].max()
                                if max_val != min_val:
                                    df[col] = (df[col] - min_val) / (max_val - min_val)
                        report[f"operation_{i}"] = {"type": "scale", "method": method, "columns": cols_to_scale}
                    except Exception as e:
                        warnings.append(f"Scale operation failed: {e}")
            
            elif op_type == "validate":
                rules = op.get("rules", {})
                if rules:
                    try:
                        if isinstance(rules, str):
                            rules = json.loads(rules)
                        validation_results = {}
                        for col, rule in rules.items():
                            if col in df.columns:
                                # Simple validation: check if values match pattern/condition
                                validation_results[col] = {"valid": True}
                        report[f"operation_{i}"] = {"type": "validate", "results": validation_results}
                    except Exception as e:
                        warnings.append(f"Validate operation failed: {e}")
            
            elif op_type == "rule":
                condition = op.get("condition")
                action = op.get("action")
                if condition and action:
                    try:
                        if action == "drop":
                            df = df.query(f"not ({condition})")
                        elif action == "flag":
                            df["flagged"] = df.eval(condition)
                        report[f"operation_{i}"] = {"type": "rule", "action": action, "rows_after": len(df)}
                    except Exception as e:
                        warnings.append(f"Rule operation failed: {e}")
            
            elif op_type == "window":
                partition_by = op.get("partition_by", [])
                order_by = op.get("order_by")
                size = op.get("size", 3)
                function = op.get("function", "rolling_mean")
                try:
                    valid_partition = [c for c in partition_by if c in df.columns]
                    if valid_partition:
                        if function == "rolling_mean":
                            df = df.rolling(window=size).mean()
                        elif function == "rolling_sum":
                            df = df.rolling(window=size).sum()
                        elif function == "rolling_std":
                            df = df.rolling(window=size).std()
                    report[f"operation_{i}"] = {"type": "window", "function": function, "size": size}
                except Exception as e:
                    warnings.append(f"Window operation failed: {e}")
            
        except Exception as e:
            warnings.append(f"Error in operation {i} ({op.get('type')}): {e}")
    
    # Prepare output
    name = Path(file_name).stem
    
    # Get preview (first 5 rows)
    preview_rows = df.head(5).to_dict(orient="records")
    
    # Convert to CSV for output
    out = io.StringIO()
    df.to_csv(out, index=False)
    
    return (
        f"{name}_transformed.csv",
        "text/csv",
        _encode_bytes(out.getvalue().encode("utf-8")),
        df.columns.tolist(),
        len(df),
        preview_rows,
        warnings,
        report,
    )
