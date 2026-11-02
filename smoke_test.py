"""End-to-end smoke test: start the MCP server and call its tools."""

from __future__ import annotations

import asyncio
import base64
import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parent
SERVER = ROOT / "paddleocr_mcp.py"
IMAGE = ROOT / "test_sample.png"


def _ensure_image() -> None:
    if IMAGE.exists():
        return
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (600, 160), "white")
    try:
        font = ImageFont.truetype("arial.ttf", 48)
    except OSError:
        font = ImageFont.truetype("DejaVuSans.ttf", 48)
    ImageDraw.Draw(image).text((30, 40), "Hello PaddleOCR 123", font=font, fill="black")
    image.save(IMAGE)



def _payload(result, label: str) -> dict:
    if result.is_error:
        raise SystemExit(f"{label} failed: {result.content[0].text}")
    text = next(item.text for item in result.content if item.type == "text")
    data = json.loads(text)
    print(f"{label}: {json.dumps(data, ensure_ascii=False)[:300]}")
    return data


async def main() -> None:
    _ensure_image()
    params = StdioServerParameters(command=sys.executable, args=[str(SERVER)], cwd=str(ROOT))

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            names = [tool.name for tool in tools.tools]
            print("tools:", names)
            assert names == ["ocr_image", "ocr_images"], names

            single = _payload(
                await session.call_tool("ocr_image", {"image_path": str(IMAGE)}),
                "ocr_image(path)",
            )
            assert "Hello PaddleOCR 123" in single["text"], single

            encoded = "data:image/png;base64," + base64.b64encode(IMAGE.read_bytes()).decode()
            encoded = encoded[:40] + "\n" + encoded[40:]
            b64 = _payload(
                await session.call_tool("ocr_image", {"image_base64": encoded}),
                "ocr_image(base64)",
            )
            assert "Hello PaddleOCR 123" in b64["text"], b64

            concurrent = await asyncio.gather(
                session.call_tool("ocr_image", {"image_path": str(IMAGE)}),
                session.call_tool("ocr_image", {"image_path": str(IMAGE)}),
            )
            for result in concurrent:
                data = _payload(result, "ocr_image(concurrent)")
                assert "Hello PaddleOCR 123" in data["text"], data

            batch = _payload(
                await session.call_tool(
                    "ocr_images", {"image_paths": [str(IMAGE), str(ROOT / "missing.png")]}
                ),
                "ocr_images",
            )
            assert batch["succeeded"] == 1 and batch["failed"] == 1, batch

    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    asyncio.run(main())
