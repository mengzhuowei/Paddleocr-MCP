# PaddleOCR MCP

把本地 PaddleOCR 封装成 MCP 服务器，让 Claude Code、Claude Desktop、Cursor、Cline 等支持 MCP 的客户端直接调用本地 OCR。

- 纯本地推理，图片不上传
- 支持中文、英文、日文、韩文等 100+ 语言
- 返回整段文本、逐行文本、置信度和坐标框
- 支持本地图片路径和 base64 图片数据
- 模型按语言懒加载并缓存，首次使用自动下载（缓存在 `~/.paddlex/official_models`）

## 环境

已在以下环境验证：

| 组件 | 版本 |
| --- | --- |
| Python | 3.13 |
| PaddleOCR | 3.7.0 |
| PaddlePaddle | 3.3.1（CPU） |
| mcp | 2.3.0 |
| 系统 | Windows 11 |

PaddleOCR 3.x 需要 `paddlepaddle` 才能推理。只安装 `paddleocr` 会在调用时报 `dependency 'paddlepaddle' is not installed`。

## 安装

```bash
python -m pip install -r requirements.txt
```

如果使用 GPU，请按 [PaddlePaddle 官方文档](https://www.paddlepaddle.org.cn/install/quick) 安装对应 CUDA 版本的 `paddlepaddle-gpu`（不要同时安装 CPU 版 `paddlepaddle`），并设置 `PADDLEOCR_MCP_DEVICE=gpu:0`。

## 测试

```bash
python smoke_test.py
```

脚本会自动生成测试图片，启动 MCP 服务器，依次验证 `tools/list`、路径识别、base64 识别和批量识别。成功时输出 `SMOKE TEST PASSED`。

也可以不用 MCP 客户端，直接验证 PaddleOCR：

```bash
PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT=0 python -c "from paddleocr import PaddleOCR; print(PaddleOCR(lang='ch').predict('test_sample.png')[0].json['res']['rec_texts'])"
```

## 配置 MCP 客户端

服务通过 **stdio** 通信，客户端把 `python` 作为子进程启动即可。请使用服务器脚本的**绝对路径**。

### Claude Code

```bash
claude mcp add -s user paddleocr -- python "E:/code/paddleocr_mcp/paddleocr_mcp.py"
```

等价的手写配置（`~/.claude.json`）：

```json
{
  "mcpServers": {
    "paddleocr": {
      "type": "stdio",
      "command": "python",
      "args": ["E:/code/paddleocr_mcp/paddleocr_mcp.py"]
    }
  }
}
```

### Claude Desktop

编辑 `%APPDATA%\Claude\claude_desktop_config.json`：

```json
{
  "mcpServers": {
    "paddleocr": {
      "command": "python",
      "args": ["E:/code/paddleocr_mcp/paddleocr_mcp.py"]
    }
  }
}
```

### Cursor / Cline

在项目的 `.cursor/mcp.json`（或 Cline 的 MCP 设置）中加入：

```json
{
  "mcpServers": {
    "paddleocr": {
      "command": "python",
      "args": ["E:/code/paddleocr_mcp/paddleocr_mcp.py"]
    }
  }
}
```

### DeepSeek Harness

DSH 用内置的 `@deepseek-ai/dsh-mcp-client` 桥接 MCP 服务器，注册项写进 profile 的
`cordis.patch.yml`：桌面版是 `~/.dsh/profiles/desktop/cordis.patch.yml`，而
`~/.dsh/cordis.patch.yml` 会对所有 profile 生效。

新增服务器必须包在 `- insert:` 列表里。直接写 `- id: ...` 的顶层条目会被当成"覆盖已有条目"，
找不到同名条目时只打印 `patch: entry "..." not found` 然后跳过，服务器不会启动。

```yaml
- insert:
    - id: mcp-paddleocr
      name: "@deepseek-ai/dsh-mcp-client"
      config:
        serverName: paddleocr
        transport: stdio
        command: python
        args:
          - E:/code/paddleocr_mcp/paddleocr_mcp.py
        cwd: E:/code/paddleocr_mcp
        toolCallTimeoutMs: 300000
        failOnStartupError: false
```

保存后 DSH 会热重载 profile 并连接服务器，工具以 `mcp__paddleocr__ocr_image`、
`mcp__paddleocr__ocr_images` 的形式出现；没有开启配置监听时需要重启 Harness。
`toolCallTimeoutMs` 设成 300000（5 分钟）是为了给 200 张图片的批量识别留足时间。

### 作为包安装

```bash
pip install -e .
```

`pyproject.toml` 不包含 `paddlepaddle`（避免覆盖 GPU 版），安装后还需自行确保推理后端已安装。之后客户端命令可以简写为 `paddleocr-mcp`。

## 工具

### `ocr_image`

识别单张图片。

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `image_path` | string | - | 本地图片路径，建议绝对路径；与 `image_base64` 二选一 |
| `image_base64` | string | - | base64 编码的图片；与 `image_path` 二选一 |
| `lang` | string | `ch` | OCR 语言，如 `ch`、`en`、`japan`、`korean`、`chinese_cht` |
| `min_confidence` | float | `0` | 最低置信度过滤，范围 0-1 |
| `include_boxes` | bool | `true` | 是否返回每行文字的坐标框 |
| `use_textline_orientation` | bool | `false` | 是否启用文本行方向分类，识别旋转文字时开启 |

返回示例：

```json
{
  "text": "Hello PaddleOCR 123",
  "lines": [
    {
      "text": "Hello PaddleOCR 123",
      "confidence": 0.9997,
      "box": [[29.0, 45.0], [506.0, 45.0], [506.0, 86.0], [29.0, 86.0]]
    }
  ],
  "count": 1,
  "source": "E:\\code\\paddleocr_mcp\\test_sample.png",
  "language": "ch",
  "elapsed_ms": 820
}
```

### `ocr_images`

批量识别多张本地图片。逐张返回结果；单张失败只会把错误写进该项的 `error`，不影响其余图片。

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `image_paths` | string[] | - | 图片路径列表，最多 200 张 |
| `lang` | string | `ch` | 同 `ocr_image` |
| `min_confidence` | float | `0` | 同 `ocr_image` |
| `include_boxes` | bool | `false` | 默认不返回坐标框，减少响应体积 |
| `use_textline_orientation` | bool | `false` | 同 `ocr_image` |

返回示例：

```json
{
  "results": [
    {"text": "第一张图的文字", "lines": ["..."], "count": 1, "source": "E:\\a.png", "language": "ch", "elapsed_ms": 780},
    {"source": "E:\\missing.png", "error": "图片不存在: E:\\missing.png (请传绝对路径 / pass an absolute path)"}
  ],
  "total": 2,
  "succeeded": 1,
  "failed": 1
}
```

## 环境变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `PADDLEOCR_MCP_LANG` | `ch` | 默认 OCR 语言 |
| `PADDLEOCR_MCP_DEVICE` | `cpu` | 推理设备，如 `cpu`、`gpu:0` |
| `PADDLEOCR_MCP_OCR_VERSION` | 由 PaddleOCR 决定 | 模型版本，如 `PP-OCRv5`、`PP-OCRv6` |
| `PADDLEOCR_MCP_PRELOAD` | `1` | 启动后在后台预热默认语言模型；设为 `0` 关闭 |

服务器启动时会自动设置：

- `PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT=0`：规避 paddlepaddle 3.3.x 在 Windows 上启用 oneDNN 时的崩溃
- `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True`：跳过启动时的模型源联网检查，加快初始化

## 工作原理

- 使用官方 `mcp` Python SDK（2.x，`MCPServer`，即原 `FastMCP`）
- 工具函数是 `async` 的，推理放在工作线程中执行，避免阻塞 MCP 事件循环
- 每个 `(语言, 设备, 方向分类开关)` 组合复用一个 OCR 引擎，模型只加载一次
- 启动时后台线程预热默认模型；首次调用如果仍在下载模型，会等待加载完成
- stdout 只用于 MCP 协议；日志和 Paddle 的警告都输出到 stderr

## 项目结构

```text
paddleocr_mcp.py    MCP 服务器与 OCR 工具
smoke_test.py       端到端测试（自动生成测试图片）
requirements.txt    Python 依赖
pyproject.toml      可安装包与 paddleocr-mcp 命令
```

## 常见问题

**`ModuleNotFoundError: No module named 'paddle'`**
只装了 `paddleocr`，还需要 `pip install paddlepaddle`。

**Windows 上报 `ConvertPirAttribute2RuntimeAttribute not support ... onednn_instruction.cc`**
paddlepaddle 3.3.x 在 Windows 上启用 oneDNN 的已知问题。本服务器已自动设置 `PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT=0`；如果手动运行 PaddleOCR，需要自己设置该变量（或改用 3.2.0）。

**第一次调用很慢**
首次运行会下载 PP-OCRv6 模型（约几十 MB），之后从 `~/.paddlex/official_models` 缓存加载。服务器启动时的预热线程会提前加载默认模型。

**找不到图片**
MCP 服务器的工作目录和客户端不一定相同，请传绝对路径。相对路径会按服务器进程的当前目录解析。

**客户端启动时报 MCP 协议或版本错误**
本项目面向 `mcp>=2.0,<3`。如果环境里是 1.x，请升级：`pip install -U "mcp>=2,<3"`。

**切换语言**
`lang` 参数按调用生效，不同语言会各自下载并缓存模型。常见别名会自动转换：`zh` → `ch`、`ja` → `japan`、`ko` → `korean`。
