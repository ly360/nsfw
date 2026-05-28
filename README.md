# NSFW Image Audit API

基于 **Falconsai / Marqo / NudeNet / LukeJacob2023** 四个模型**投票**判定的图片内容审计 HTTP API。每个模型贡献一票，**≥ 2 票才判 NSFW**，单点误报会被其他模型否决。

## 模型说明

启动时一次性加载，常驻内存。各自定位、长板与已知短板：

| 模型 | 类型 | 输出 | 擅长 | 已知短板 |
|---|---|---|---|---|
| **Falconsai/nsfw_image_detection** | ViT-base 二分类 | `nsfw` / `sfw` 概率 | 真人裸露照片 | 色彩偏置严重——粉/红/肉色为主的游戏皮肤、樱花、化妆品广告等极易误报 |
| **Marqo/nsfw-image-detection-384** | timm 二分类（384 分辨率） | `nsfw` / `sfw` 概率 | 比 Falconsai 更新更稳的二分类基线 | 对"穿着但暗示"类图片偏向 SFW |
| **NudeNet** | ONNX 部位检测 | 多个 `(class, box, score)` | 显式暴露部位（生殖器/乳房/臀部/肛门）的定位 | 只看真人，对动漫/3D 渲染的紧身衣可能把裆部纹理框成 `FEMALE_GENITALIA_EXPOSED`；对穿着暗示场景无能为力 |
| **LukeJacob2023/nsfw-image-detector** | ViT-base 5 分类 | `drawings` / `hentai` / `neutral` / `porn` / `sexy` 概率 | "穿着但性感"（短裙/比基尼/挑逗姿势）、动漫 hentai 与 SFW drawings 的区分 | 真人/动漫的边界偶尔模糊；阈值需根据业务调 |

## 判定逻辑

参考 `app/audit.py:combine`：

```
vote 1: Falconsai nsfw           ≥ NSFW_THRESHOLD
vote 2: Marqo nsfw               ≥ NSFW_THRESHOLD
vote 3: NudeNet 暴露部位         有任一框 ≥ NUDENET_DETECT_THRESHOLD（限于 EXPOSED 集）
vote 4: Luke hentai+porn+sexy 之和 ≥ LUKE_THRESHOLD

verdict = NSFW if len(votes) >= 2 else SAFE
```

`reason` 字段会展示触发的投票：

- 0 票：`"-"`
- 1 票（被否决）：`"单票忽略: Falconsai=0.99"`，便于日志中观察单点误报的模型
- ≥ 2 票：`"Marqo=0.68 | NudeNet=FEMALE_BREAST_EXPOSED"`

## 环境

Python 已通过 venv 安装在仓库根目录。**始终使用 `./bin/python` 调用解释器**（直接用 `./bin/pip` 会因 shebang 失效报错；安装包请用 `./bin/python -m pip`）。

依赖已安装：`fastapi`、`uvicorn`、`python-multipart`、`torch`、`timm`、`transformers`、`nudenet`、`onnxruntime`、`pillow`。

## 启动服务

```bash
./bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

启动时会一次性加载**四个模型**到内存（首次会从 HuggingFace 拉权重，之后走本地缓存）。日志会打印自动选中的设备（CUDA / MPS / CPU）和 ONNX providers。

打开 `http://127.0.0.1:8000/docs` 可以看自动生成的 Swagger UI。

## 配置（环境变量）

| 变量 | 默认 | 含义 |
|---|---|---|
| `NSFW_THRESHOLD` | `0.5` | Falconsai / Marqo 的 NSFW 概率阈值 |
| `NUDENET_DETECT_THRESHOLD` | `0.5` | NudeNet 单个检测框的最低置信度（值越低越敏感，0.3 容易把动漫紧身衣误判） |
| `LUKE_THRESHOLD` | `0.5` | Luke 模型 `hentai + porn + sexy` 三类概率之和的阈值 |

例：

```bash
NSFW_THRESHOLD=0.7 LUKE_THRESHOLD=0.6 ./bin/python -m uvicorn app.main:app --port 8000
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
  "reason": "Marqo=0.68 | Luke=0.92(porn=0.81)",
  "scores": {
    "falconsai_nsfw": 0.0036,
    "marqo_nsfw": 0.6811,
    "nudenet_top_score": 0.0,
    "nudenet_labels": [],
    "luke_nsfw_score": 0.9214,
    "luke_scores": {
      "drawings": 0.0021,
      "hentai": 0.0512,
      "neutral": 0.0765,
      "porn": 0.8108,
      "sexy": 0.0594
    }
  },
  "elapsed_ms": 412
}
```

字段说明：

- `verdict`：`NSFW` 或 `SAFE`
- `reason`：触发投票的清单，例如 `Falconsai=0.92 | NudeNet=FEMALE_BREAST_EXPOSED`；`SAFE` 且单票时为 `单票忽略: Falconsai=0.99`；0 票时为 `-`
- `scores.falconsai_nsfw` / `scores.marqo_nsfw`：两个二分类模型的 NSFW 概率
- `scores.nudenet_top_score`：NudeNet 命中"暴露部位"中的最高置信度，没命中则为 `0.0`
- `scores.nudenet_labels`：命中的暴露类别，如 `["FEMALE_BREAST_EXPOSED"]`
- `scores.luke_nsfw_score`：Luke 模型 `hentai + porn + sexy` 三类概率之和（值越大越像不良内容；与 `LUKE_THRESHOLD` 对比）
- `scores.luke_scores`：Luke 模型 5 个类的完整概率明细，方便做后续分析或调整投票口径
- `elapsed_ms`：服务端四模型推理总耗时（不含网络 / 文件读取）

NudeNet 监控的暴露类别（在 `app/audit.py:NUDENET_EXPOSED_CLASSES` 中可改）：

```
FEMALE_GENITALIA_EXPOSED, MALE_GENITALIA_EXPOSED,
FEMALE_BREAST_EXPOSED, BUTTOCKS_EXPOSED, ANUS_EXPOSED
```

Luke 用于投票的"不良"类别（在 `app/audit.py:LUKE_NSFW_LABELS` 中可改）：

```
hentai, porn, sexy
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
├── models.py     # ModelBundle：四个模型的加载与推理封装
├── audit.py      # 投票判定 + 阈值（环境变量）
└── schemas.py    # 请求/响应 Pydantic 结构
test_nsfw.py      # 离线对比脚本（保留，不参与 API）
pic/              # 测试图片
```

## 已知限制与调优建议

- **资源**：单进程内存常驻**四个**模型（三个 PyTorch + 一个 ONNX），单机服务起步；多 worker 会成倍占内存。
- **吞吐**：推理同步阻塞，单机约个位数 QPS，取决于设备（CUDA > MPS > CPU）。
- **Falconsai 噪音**：粉/红/肉色主导的非裸露图片容易被它单独打高分；2-of-N 投票已经能抑制单点误报，但如果观察日志发现 Falconsai 长期"独投错票零救场"，可以考虑直接把它从投票里摘掉，只保留打分。
- **NudeNet 阈值**：默认 `0.5` 偏稳；调到 `0.3` 时容易把动漫紧身衣裆部纹理误识为 `FEMALE_GENITALIA_EXPOSED`。
- **Luke 阈值**：`hentai + porn + sexy` 之和默认 `0.5`。如果业务需要拦截"穿着但暗示"图片（如短裙/泳装写真），可以把阈值降到 `0.3` 或让 Luke 单票即定罪；但会显著拉高误报率，建议先观察日志再调。
- **覆盖盲区**：露背、透视装、暴力 / 血腥、文字辱骂、未成年人识别都不在当前栈覆盖范围内。
