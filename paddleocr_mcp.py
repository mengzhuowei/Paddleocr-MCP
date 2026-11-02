"""Local PaddleOCR exposed as an MCP server over stdio."""

from __future__ import annotations

import asyncio
import base64
import binascii
import io
import os
import sys
import threading
import time
from pathlib import Path
from typing import Annotated, Any

# Read by PaddleX before paddle is imported: skip the model-source connectivity
# check, and avoid the paddlepaddle 3.3.x Windows crash on PP-OCR inference.
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "0")
os.environ.setdefault("GLOG_minloglevel", "2")

from mcp.server.mcpserver import MCPServer
from pydantic import Field

__version__ = "0.1.0"

DEFAULT_LANG = os.environ.get("PADDLEOCR_MCP_LANG", "ch")
DEFAULT_DEVICE = os.environ.get("PADDLEOCR_MCP_DEVICE", "cpu")
OCR_VERSION = os.environ.get("PADDLEOCR_MCP_OCR_VERSION") or None
PRELOAD = os.environ.get("PADDLEOCR_MCP_PRELOAD", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}

_LANG_ALIASES = {
    "zh": "ch",
    "zh-cn": "ch",
    "cn": "ch",
    "zh-tw": "chinese_cht",
    "cht": "chinese_cht",
    "ja": "japan",
    "jp": "japan",
    "ko": "korean",
    "kr": "korean",
}

_engines: dict[tuple[str, str, bool], Any] = {}
_engine_locks: dict[tuple[str, str, bool], threading.Lock] = {}
_engine_lock = threading.Lock()


def _log(message: str) -> None:
    print(f"[paddleocr-mcp] {message}", file=sys.stderr, flush=True)


def _normalize_lang(lang: str) -> str:
    value = (lang or DEFAULT_LANG).strip()
    return _LANG_ALIASES.get(value.lower(), value)


def _get_engine(
    lang: str, device: str, use_textline_orientation: bool
) -> tuple[Any, threading.Lock]:
    key = (lang, device, use_textline_orientation)
    with _engine_lock:
        engine = _engines.get(key)
        if engine is None:
            from paddleocr import PaddleOCR

            _log(f"loading PaddleOCR (lang={lang}, device={device})...")
            started = time.perf_counter()
            engine = PaddleOCR(
                lang=lang,
                device=device,
                ocr_version=OCR_VERSION,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=use_textline_orientation,
            )
            _engines[key] = engine
            _engine_locks[key] = threading.Lock()
            _log(f"PaddleOCR ready in {time.perf_counter() - started:.1f}s")
        return engine, _engine_locks[key]


def _to_float(value: Any) -> float | None:
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None


def _to_box(poly: Any) -> list[list[float]] | None:
    if poly is None:
        return None
    try:
        return [[round(float(x), 2), round(float(y), 2)] for x, y in poly]
    except (TypeError, ValueError):
        return None


def _page_payload(item: Any, include_boxes: bool, min_confidence: float) -> dict[str, Any]:
    payload = getattr(item, "json", item)
    if isinstance(payload, dict):
        payload = payload.get("res", payload)
    if not isinstance(payload, dict):
        return {"text": "", "lines": [], "count": 0}

    texts = list(payload.get("rec_texts") or [])
    scores = list(payload.get("rec_scores") or [])
    polys = payload.get("rec_polys")
    if polys is None:
        polys = payload.get("dt_polys")
    polys = list(polys) if polys is not None else []

    lines: list[dict[str, Any]] = []
    for index, text in enumerate(texts):
        confidence = _to_float(scores[index]) if index < len(scores) else None
        if confidence is not None and confidence < min_confidence:
            continue
        line: dict[str, Any] = {"text": str(text)}
        if confidence is not None:
            line["confidence"] = confidence
        if include_boxes and index < len(polys):
            box = _to_box(polys[index])
            if box:
                line["box"] = box
        lines.append(line)

    return {
        "text": "\n".join(line["text"] for line in lines),
        "lines": lines,
        "count": len(lines),
    }


def _resolve_path(image_path: str) -> Path:
    path = Path(image_path).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if not path.exists():
        raise ValueError(f"图片不存在: {path} (请传绝对路径 / pass an absolute path)")
    if not path.is_file():
        raise ValueError(f"不是文件: {path}")
    return path


def _image_from_base64(data: str) -> Any:
    payload = data.strip()
    if payload.startswith("data:") and "," in payload:
        payload = payload.split(",", 1)[1]
    try:
        raw = base64.b64decode("".join(payload.split()), validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("image_base64 不是合法的 base64 数据") from exc
    try:
        from PIL import Image
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("解码 base64 图片需要 Pillow 和 numpy") from exc
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise ValueError(f"无法解码图片: {exc}") from exc
    return np.array(image)


def _run_ocr(
    *,
    image_path: str | None,
    image_base64: str | None,
    lang: str,
    min_confidence: float,
    include_boxes: bool,
    use_textline_orientation: bool,
) -> dict[str, Any]:
    if bool(image_path) == bool(image_base64):
        raise ValueError("image_path 和 image_base64 必须且只能提供一个")

    source: str
    if image_path:
        path = _resolve_path(image_path)
        source = str(path)
        image: Any = source
    else:
        source = "<base64>"
        image = _image_from_base64(image_base64 or "")

    language = _normalize_lang(lang)
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError("min_confidence 必须在 0 到 1 之间")

    started = time.perf_counter()
    try:
        engine, infer_lock = _get_engine(language, DEFAULT_DEVICE, use_textline_orientation)
        with infer_lock:
            results = engine.predict(image)
    except Exception as exc:
        raise RuntimeError(
            f"OCR 失败: {exc}\n"
            "请确认已安装 paddlepaddle (pip install paddlepaddle)，"
            f"并且 lang={language!r} 是 PaddleOCR 支持的语言。"
        ) from exc

    pages = [_page_payload(item, include_boxes, min_confidence) for item in (results or [])]
    if not pages:
        pages = [{"text": "", "lines": [], "count": 0}]
    if len(pages) == 1:
        payload = pages[0]
    else:
        texts = [page["text"] for page in pages if page["text"]]
        payload = {
            "text": "\n\n".join(texts),
            "lines": [line for page in pages for line in page["lines"]],
            "count": sum(page["count"] for page in pages),
        }

    payload.update(
        {
            "source": source,
            "language": language,
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
        }
    )
    return payload


mcp = MCPServer(
    name="paddleocr",
    title="PaddleOCR",
    version=__version__,
    description="本地 PaddleOCR 文字识别 / Local OCR with PaddleOCR",
    instructions=(
        "使用 ocr_image 识别单张图片中的文字，使用 ocr_images 批量识别多张图片。"
        "支持中文、英文等 100 多种语言；图片必须是服务器可访问的本地文件路径，"
        "也可以传 base64 编码的图片数据。"
    ),
)


@mcp.tool()
async def ocr_image(
    image_path: Annotated[
        str | None,
        Field(description="本地图片路径（推荐绝对路径）；与 image_base64 二选一"),
    ] = None,
    image_base64: Annotated[
        str | None,
        Field(description="Base64 编码的图片数据；与 image_path 二选一"),
    ] = None,
    lang: Annotated[
        str,
        Field(description="OCR 语言，如 ch（中英）、en、japan、korean，默认 ch"),
    ] = DEFAULT_LANG,
    min_confidence: Annotated[
        float,
        Field(description="最低置信度过滤，0-1，默认 0 表示不过滤"),
    ] = 0.0,
    include_boxes: Annotated[
        bool,
        Field(description="是否返回每行文字的坐标框"),
    ] = True,
    use_textline_orientation: Annotated[
        bool,
        Field(description="是否启用文本行方向分类（识别旋转文字时开启，速度略慢）"),
    ] = False,
) -> dict[str, Any]:
    """识别单张图片中的文字。返回完整文本、逐行结果、置信度和坐标框。

    Recognize text in one local image (file path or base64). Returns the full
    text, per-line text with confidence, and optional bounding boxes.
    """
    return await asyncio.to_thread(
        _run_ocr,
        image_path=image_path,
        image_base64=image_base64,
        lang=lang,
        min_confidence=min_confidence,
        include_boxes=include_boxes,
        use_textline_orientation=use_textline_orientation,
    )


@mcp.tool()
async def ocr_images(
    image_paths: Annotated[
        list[str],
        Field(description="本地图片路径列表，建议使用绝对路径"),
    ],
    lang: Annotated[
        str,
        Field(description="OCR 语言，如 ch（中英）、en、japan、korean，默认 ch"),
    ] = DEFAULT_LANG,
    min_confidence: Annotated[
        float,
        Field(description="最低置信度过滤，0-1，默认 0 表示不过滤"),
    ] = 0.0,
    include_boxes: Annotated[
        bool,
        Field(description="是否返回每行文字的坐标框"),
    ] = False,
    use_textline_orientation: Annotated[
        bool,
        Field(description="是否启用文本行方向分类（识别旋转文字时开启，速度略慢）"),
    ] = False,
) -> dict[str, Any]:
    """批量识别多张本地图片中的文字。逐张返回结果，单张失败不影响其余图片。

    Recognize text in multiple local images. Each result contains the same fields
    as ocr_image; a failure on one image is reported in that item's "error".
    """
    if not image_paths:
        raise ValueError("image_paths 不能为空")
    if len(image_paths) > 200:
        raise ValueError("一次最多识别 200 张图片")

    results: list[dict[str, Any]] = []
    for image_path in image_paths:
        try:
            results.append(
                await asyncio.to_thread(
                    _run_ocr,
                    image_path=image_path,
                    image_base64=None,
                    lang=lang,
                    min_confidence=min_confidence,
                    include_boxes=include_boxes,
                    use_textline_orientation=use_textline_orientation,
                )
            )
        except (ValueError, RuntimeError) as exc:
            results.append({"source": image_path, "error": str(exc)})

    succeeded = sum(1 for item in results if "error" not in item)
    return {
        "results": results,
        "total": len(results),
        "succeeded": succeeded,
        "failed": len(results) - succeeded,
    }


def _warmup() -> None:
    try:
        _get_engine(_normalize_lang(DEFAULT_LANG), DEFAULT_DEVICE, False)
    except Exception as exc:
        _log(f"模型预热失败（首次调用时会重试）: {exc}")


def main() -> None:
    if PRELOAD:
        threading.Thread(target=_warmup, name="paddleocr-warmup", daemon=True).start()
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
