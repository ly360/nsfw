# NSFW Image Audit API

基于 **Falconsai / Marqo / NudeNet** 三模型综合判定的图片内容审计 HTTP API。任一模型命中即判为 NSFW。

## 环境

Python 已通过 venv 安装在仓库根目录。**始终使用 `./bin/python` 调用解释器**（直接用 `./bin/pip` 会因 shebang 失效报错；安装包请用 `./bin/python -m pip`）。

依赖已安装：`fastapi`、`uvicorn`、`python-multipart`、`torch`、`timm`、`transformers`、`nudenet`、`onnxruntime`、`pillow`。

## 启动服务

```bash
./bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

启动时会一次性加载三个模型到内存（首次会从 HuggingFace 拉权重，之后走本地缓存）。日志会打印自动选中的设备（CUDA / MPS / CPU）和 ONNX providers。

打开 `http://127.0.0.1:8000/docs` 可以看自动生成的 Swagger UI。

## 配置（环境变量）

| 变量 | 默认 | 含义 |
|---|---|---|
| `NSFW_THRESHOLD` | `0.5` | Falconsai / Marqo 的 NSFW 概率阈值 |
| `NUDENET_DETECT_THRESHOLD` | `0.3` | NudeNet 单个检测框的最低置信度 |

例：

```bash
NSFW_THRESHOLD=0.7 ./bin/python -m uvicorn app.main:app --port 8000
```

## 接口

### `GET /healthz`

探活，返回设备和 ONNX providers。

```bash
curl -s http://127.0.0.1:8000/healthz
```

```json
{"status":"ok","device":"mps","onnx_providers":["CoreMLExecutionProvider","CPUExecutionProvider"]}
```

### `POST /audit/image`

`multipart/form-data` 上传单张图片审计。

- 字段名：`file`
- 支持类型：`image/jpeg` / `image/png` / `image/webp`
- 单文件上限：10 MB

```bash
curl -s -X POST http://127.0.0.1:8000/audit/image \
  -F "file=@pic/1.jpg;type=image/jpeg" | python3 -m json.tool
```

响应：

```json
{
  "verdict": "NSFW",
  "reason": "Marqo=0.68",
  "scores": {
    "falconsai_nsfw": 0.0036,
    "marqo_nsfw": 0.6811,
    "nudenet_top_score": 0.0,
    "nudenet_labels": []
  },
  "elapsed_ms": 317
}
```

字段说明：

- `verdict`：`NSFW` 或 `SAFE`
- `reason`：触发原因，例如 `Falconsai=0.92 | NudeNet=FEMALE_BREAST_EXPOSED`；`SAFE` 时为 `-`
- `scores.falconsai_nsfw` / `scores.marqo_nsfw`：两个分类模型给出的 NSFW 概率
- `scores.nudenet_top_score`：NudeNet 命中"暴露部位"中的最高置信度，没命中则为 `0.0`
- `scores.nudenet_labels`：命中的暴露类别，如 `["FEMALE_BREAST_EXPOSED"]`
- `elapsed_ms`：服务端三模型推理总耗时（不含网络 / 文件读取）

NudeNet 监控的暴露类别（在 `app/audit.py` 中可改）：

```
FEMALE_GENITALIA_EXPOSED, MALE_GENITALIA_EXPOSED,
FEMALE_BREAST_EXPOSED, BUTTOCKS_EXPOSED, ANUS_EXPOSED
```

### 错误响应

| HTTP | 触发条件 | 示例 |
|---|---|---|
| 400 | 文件为空 / 超 10MB / 无法解析为图片 | `{"detail":"无法解析的图片格式"}` |
| 415 | content-type 不在白名单 | `{"detail":"不支持的 content-type: application/octet-stream, ..."}` |
| 422 | 缺 `file` 字段 | FastAPI 标准校验错误 |

## curl 示例集合

```bash
# 1. 单张图（带显式 content-type，最稳）
curl -s -X POST http://127.0.0.1:8000/audit/image \
  -F "file=@pic/1.jpg;type=image/jpeg"

# 2. PNG 截图
curl -s -X POST http://127.0.0.1:8000/audit/image \
  -F "file=@pic/ScreenShot_2026-05-27_162714_118.jpg;type=image/jpeg"

# 3. 只看判定字段
curl -s -X POST http://127.0.0.1:8000/audit/image \
  -F "file=@pic/1.jpg;type=image/jpeg" \
  | python3 -c "import sys,json; r=json.load(sys.stdin); print(r['verdict'], '|', r['reason'])"

# 4. 批量遍历 pic/ 目录
for f in pic/*.jpg; do
  printf "%-50s " "$f"
  curl -s -X POST http://127.0.0.1:8000/audit/image -F "file=@$f;type=image/jpeg" \
    | python3 -c "import sys,json; r=json.load(sys.stdin); print(r['verdict'], r['reason'])"
done

# 5. 看 HTTP 状态码（错误调试）
curl -s -o /dev/null -w "HTTP %{http_code}\n" \
  -X POST http://127.0.0.1:8000/audit/image \
  -F "file=@README.md;type=image/jpeg"
```

## 目录结构

```
app/
├── __init__.py
├── main.py       # FastAPI 入口、lifespan 加载模型、路由
├── models.py     # ModelBundle：三模型的加载与推理封装
├── audit.py      # 综合判定 + 阈值（环境变量）
└── schemas.py    # 请求/响应 Pydantic 结构
test_nsfw.py      # 离线对比脚本（保留，不参与 API）
pic/              # 测试图片
```

## 已知限制

- 单进程内存常驻三个模型，**单机服务**起步，多 worker 会成倍占内存。
- 推理是同步阻塞，PyTorch 模型本身串行，单机吞吐约个位数 QPS（取决于设备）。
- NudeNet 类别和阈值是粗粒度，露背 / 透视 / 卡通等场景可能漏判；如需更严格，调高 Falconsai/Marqo 灵敏度或改为加权打分（见方案中"演进"小节）。
