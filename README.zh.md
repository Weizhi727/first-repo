# SSVP：工業零樣本異常檢測的協同語義視覺提示框架

> **SSVP**（arXiv 2601.09147）的程式碼實作——一個零樣本異常檢測框架，透過三個緊密耦合的模組，將 CLIP 語義表示與 DINOv2 細粒度結構特徵融合。

---

## 研究背景

現有的 ZSAD（零樣本異常檢測）方法受限於單一視覺主幹，在全局語義泛化能力與細粒度結構判別力之間難以取得平衡。SSVP 透過高效融合兩個互補編碼器來解決這一問題：

| 模組 | 功能 |
|------|------|
| **HSVS**（層次語義視覺協同） | 透過雙路徑跨模態注意力（ATF Block），將 DINOv2 多尺度結構先驗注入 CLIP 語義流形 |
| **VCPG**（視覺條件提示生成器） | 使用 VAE 式編碼器與跨模態注意力，引導文字嵌入精確錨定於缺陷區域 |
| **VTAM**（視覺文本異常映射器） | 採用可微分軟門控的混合專家（MoE），動態以局部區塊證據校準全局分數 |

CLIP 與 DINOv2 主幹**完全凍結**，只有三個 SSVP 頭部模組參與訓練。

### MVTec-AD 零樣本實驗結果

| 方法 | Image-AUROC | Pixel-AUROC |
|------|:-----------:|:-----------:|
| WinCLIP | 85.1 | 85.1 |
| AnomalyCLIP | 91.5 | 91.1 |
| VCP-CLIP | 91.8 | 91.7 |
| Bayes-PFL | 92.4 | 91.9 |
| **SSVP（本專案）** | **93.0** | **92.2** |

---

## 專案結構

```
first-repo/
└── ssvp/
    ├── models/
    │   ├── __init__.py
    │   ├── hsvs.py       # 層次語義視覺協同模組
    │   ├── vcpg.py       # 視覺條件提示生成器
    │   ├── vtam.py       # 視覺文本異常映射器 + MoE
    │   └── ssvp.py       # 頂層 SSVP 模型與 SSVPConfig
    ├── utils/
    │   ├── __init__.py
    │   └── anomaly_map.py
    ├── train.py          # 完整訓練流程
    ├── inference.py      # 推論 CLI 與 Python API
    └── requirements.txt
```

---

## 安裝

```bash
git clone https://github.com/<your-username>/first-repo.git
cd first-repo

pip install -r ssvp/requirements.txt

# 安裝 OpenAI CLIP
pip install git+https://github.com/openai/CLIP.git

# DINOv2 透過 torch.hub 載入，需要 timm
pip install timm>=0.9.0
```

**主要依賴套件版本：**

| 套件 | 版本 |
|------|------|
| torch | ≥ 2.1.0 |
| torchvision | ≥ 0.16.0 |
| openai-clip | ≥ 1.0.1 |
| timm | ≥ 0.9.0 |
| Pillow | ≥ 10.0.0 |
| scikit-learn | ≥ 1.3.0 |
| numpy | ≥ 1.24.0 |
| scipy | ≥ 1.11.0 |

---

## 訓練資料集準備

SSVP 採用標準 **MVTec-AD** 資料夾格式。訓練集只需要正常圖片——合成異常圖片由 Perlin 噪聲增強器在訓練時即時生成，不需要手動準備。

```
dataset_root/
└── <category>/               # 例如 "bottle"、"transistor"
    ├── train/
    │   └── good/             # 僅需放置正常品圖片
    │       ├── 000.png
    │       └── ...
    ├── test/
    │   ├── good/             # 正常品測試圖
    │   ├── broken_large/     # 每種缺陷類型一個子目錄
    │   └── contamination/
    └── ground_truth/         # 像素級二值遮罩（用於 Pixel-AUROC 評測）
        ├── broken_large/
        │   └── 000_mask.png  # 對應 test/broken_large/000.png
        └── contamination/
```

**自訂工業資料集：** 依照相同結構建立自己的類別目錄。在 `train/good/` 中收集正常品圖片（建議 ≥ 200 張），並將測試圖片與對應遮罩分別放入 `test/` 和 `ground_truth/`。

---

## 訓練

```bash
python ssvp/train.py \
    --data-root  /path/to/mvtec \
    --category   bottle \
    --clip-ckpt  /path/to/ViT-B-16.pt \
    --dino-ckpt  /path/to/dinov2_vitb14.pth \
    --output-dir checkpoints/
```

### 完整參數說明

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--data-root` | — | 資料集根目錄 |
| `--category` | — | 類別子目錄名稱 |
| `--image-size` | `518` | 輸入解析度（正方形） |
| `--epochs` | `10` | 訓練週期數 |
| `--batch-size` | `16` | 批次大小 |
| `--lr` | `4e-5` | AdamW 學習率 |
| `--tau` | `0.07` | 對齊損失的溫度參數 τ |
| `--margin` | `0.30` | 餘弦邊距閾值 |
| `--lambda-pixel` | `0.50` | 像素層級 Focal-BCE 損失權重 |
| `--lambda-kl` | `1e-4` | VAE KL 散度損失權重 |
| `--lambda-margin` | `0.10` | 邊距損失權重 |
| `--clip-model` | `ViT-B/16` | CLIP 變體（用於推斷特徵維度） |
| `--dino-model` | `dinov2_vitb14` | DINOv2 架構變體 |
| `--output-dir` | `checkpoints/` | 模型儲存路徑 |

### 損失函數

```
L_total = L_align + λ_pixel · L_pixel + λ_kl · L_kl + λ_margin · L_margin
```

| 損失項 | 說明 |
|--------|------|
| `L_align` | 溫度 τ 縮放的圖像層級 [正常, 異常] logit 交叉熵 |
| `L_pixel` | 預測異常圖與合成 GT 遮罩之間的 Focal-BCE |
| `L_kl` | VCPG 的 VAE KL 散度（防止表示崩塌） |
| `L_margin` | 正常/異常餘弦相似度違反邊距約束時的懲罰項 |

訓練結束後，以下兩個 checkpoint 會儲存到 `--output-dir`：
- `ssvp_<category>_best.pt` — Pixel-AUROC 最高時自動儲存
- `ssvp_<category>_last.pt` — 最後一個 epoch

Checkpoint 只包含 HSVS + VCPG + VTAM 的權重，不包含凍結的主幹網路，因此體積較小。

---

## 推論

### 命令列

```bash
# 單張圖片
python ssvp/inference.py \
    --images     path/to/image.jpg \
    --class-name bottle \
    --checkpoint checkpoints/ssvp_bottle_best.pt \
    --clip-ckpt  models/ViT-B-16.pt \
    --dino-ckpt  models/dinov2_vitb14.pth \
    --output-dir results/

# 批次處理（自訂閾值）
python ssvp/inference.py \
    --images     test/images/*.jpg \
    --class-name transistor \
    --threshold  0.5 \
    --checkpoint checkpoints/ssvp_transistor_best.pt \
    --output-dir results/
```

每張圖片會在 `--output-dir` 產生對應的 `<檔名>_anomaly_map.png` 熱力圖。

### Python API

```python
import torch
from PIL import Image
from ssvp.inference import build_model, build_transforms, infer_image
from ssvp.utils import apply_colormap

device = torch.device("cuda")

# 載入模型（只需初始化一次）
model = build_model(
    clip_name="ViT-B/16",
    dino_name="dinov2_vitb14",
    checkpoint="checkpoints/ssvp_bottle_best.pt",
    device=device,
    clip_ckpt="models/ViT-B-16.pt",
    dino_ckpt="models/dinov2_vitb14.pth",
)
clip_tf, dino_tf = build_transforms(image_size=518)

# 推論單張圖片
pil_img = Image.open("test.jpg").convert("RGB")
result = infer_image(model, pil_img, class_name="bottle",
                     clip_tf=clip_tf, dino_tf=dino_tf,
                     device=device, image_size=518)

print(f"異常分數：{result['score']:.4f}")
# result["anomaly_map"] → torch.Tensor (518, 518)，值越高代表越異常

# 儲存熱力圖
colored = apply_colormap(result["anomaly_map"], colormap="jet")  # (H, W, 3) uint8
Image.fromarray(colored).save("heatmap.png")
```

### 推論輸出說明

| 輸出 Key | 型別 | 說明 |
|----------|------|------|
| `score` | `float` | 圖像層級異常分數，越高代表越異常 |
| `anomaly_map` | `Tensor (H, W)` | 像素層級異常熱力圖，值域 [0, 1] |

> **閾值選擇建議：** `--threshold 0.5` 為預設值，實際使用時應根據驗證集的 ROC 曲線選擇使 F1-score 最大的最佳閾值。

---

## 模型架構

```
輸入圖片 (B, 3, H, W)
        │
        ├──[凍結]──► CLIP ViT ──► patch tokens  (B, N, 512)
        │
        └──[凍結]──► DINOv2  ──► 多尺度 tokens  3 × (B, N, 768)
                                          │
                               ┌──────────▼──────────┐
                               │        HSVS          │
                               │  3 個 ATF Block      │
                               │  雙路徑跨模態注意力   │
                               └──────────┬──────────┘
                                          │ 增強視覺特徵 (B, N, 512)
                               ┌──────────▼──────────┐
                               │        VCPG          │
                               │  VAE 編碼器           │
                               │  跨模態注意力         │
                               └──────┬───────┬───────┘
                           條件文字    │       │ KL 損失
                                      │
                               ┌──────▼──────────────┐
                               │        VTAM          │
                               │  MoE 軟門控           │
                               │  局部-全局融合        │
                               └──────────┬──────────┘
                                          │
                        異常分數 (B,)  +  異常圖 (B, H, W)
```

---

## 引用

```bibtex
@article{ssvp2026,
  title   = {SSVP: Synergistic Semantic-Visual Prompting for Industrial Zero-Shot Anomaly Detection},
  journal = {arXiv preprint arXiv:2601.09147},
  year    = {2026},
}
```

---

## 致謝

- [OpenAI CLIP](https://github.com/openai/CLIP)
- [DINOv2 — facebookresearch](https://github.com/facebookresearch/dinov2)
- [AnomalyCLIP](https://github.com/zqhang/AnomalyCLIP) 與 [VCP-CLIP](https://github.com/xiaozhen228/VCP-CLIP) 的方法論參考
- [MVTec AD 資料集](https://www.mvtec.com/company/research/datasets/mvtec-ad)

---

> 英文版 README 請參閱 [README.md](./README.md)
