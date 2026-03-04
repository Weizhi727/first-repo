# SSVP：工業零樣本異常檢測的協同語義視覺提示框架

> **SSVP**（arXiv 2601.09147）的程式碼實作——一個零樣本異常檢測框架，透過三個緊密耦合的模組，將 CLIP 語義表示與 **DINOv3** 細粒度結構特徵融合。

---

## 研究背景

現有的 ZSAD（零樣本異常檢測）方法受限於單一視覺主幹，在全局語義泛化能力與細粒度結構判別力之間難以取得平衡。SSVP 透過高效融合兩個互補編碼器來解決這一問題：

| 模組 | 功能 |
|------|------|
| **HSVS**（層次語義視覺協同） | 透過雙路徑跨模態注意力（ATF Block），將 DINOv3 多尺度結構先驗注入 CLIP 語義流形 |
| **VCPG**（視覺條件提示生成器） | 使用 VAE 式編碼器與跨模態注意力，引導文字嵌入精確錨定於缺陷區域 |
| **VTAM**（視覺文本異常映射器） | 採用可微分軟門控的混合專家（MoE），動態以局部區塊證據校準全局分數 |

CLIP 與 DINOv3 主幹**完全凍結**，只有三個 SSVP 頭部模組（HSVS、VCPG、VTAM）參與訓練。

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
├── README.md          # 英文說明文件
├── README.zh.md       # 中文說明文件（本文件）
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

# 安裝主要依賴
pip install -r ssvp/requirements.txt

# 安裝 OpenAI CLIP
pip install git+https://github.com/openai/CLIP.git

# DINOv3 透過 torch.hub 載入，需要 timm；HuggingFace 格式需要 transformers
pip install timm>=0.9.0 transformers>=4.40.0
```

**主要依賴套件：**

| 套件 | 版本 |
|------|------|
| torch | ≥ 2.7.1 |
| torchvision | ≥ 0.16.0 |
| openai-clip | ≥ 1.0.1 |
| timm | ≥ 0.9.0 |
| transformers | ≥ 4.40.0 |
| Pillow | ≥ 10.0.0 |
| scikit-learn | ≥ 1.3.0 |
| numpy | ≥ 1.24.0 |
| scipy | ≥ 1.11.0 |

---

## 模型權重準備

### CLIP 權重

從 HuggingFace 下載 `openai/clip-vit-base-patch16` 的所有檔案，儲存到本地資料夾：

```
clip-weight/          ← 整個資料夾路徑傳給 --clip-ckpt
├── config.json
├── model.safetensors  (或 pytorch_model.bin)
├── preprocessor_config.json
├── tokenizer.json
└── vocab.json
```

> 程式會自動偵測是**資料夾（HuggingFace 格式）**還是**單一 `.pt` 檔案（OpenAI 原版格式）**，兩種格式都支援。

### DINOv3 權重

將下載好的 `dinov3_vitb16_pretrain_lvd1689m.pth` 放到本地資料夾：

```
dino-weight/
└── dinov3_vitb16_pretrain_lvd1689m.pth   ← 傳給 --dino-ckpt
```

---

## 資料集準備

SSVP 採用 **MVTec-AD 相容格式**。訓練集只需要正常圖片——合成異常圖片由 Perlin 噪聲增強器在訓練時即時生成，**不需要手動準備異常樣本**。

### 方案一：MVTec-AD（直接使用）

從 [MVTec 官方頁面](https://www.mvtec.com/company/research/datasets/mvtec-ad) 下載，解壓縮後目錄結構如下：

```
mvtec/
└── bottle/
    ├── train/
    │   └── good/             ← 正常訓練圖片（直接使用）
    │       ├── 000.png
    │       └── ...
    ├── test/
    │   ├── good/             ← 正常品測試圖
    │   ├── broken_large/     ← 每種缺陷類型一個子目錄
    │   └── contamination/
    └── ground_truth/
        ├── broken_large/
        │   └── 000_mask.png  ← 像素級二值遮罩
        └── contamination/
```

MVTec-AD 共 **15 個類別**：bottle、cable、capsule、carpet、grid、hazelnut、leather、metal_nut、pill、screw、tile、toothbrush、transistor、wood、zipper。

---

### 方案二：VisA 資料集（用 spot-diff-main prepare_data 轉換）

VisA（Visual Anomaly）共 **12 個類別**、10,821 張高解析度圖片，覆蓋電路板、食品、工業零件等場景。

#### 步驟 1：下載原始 VisA 資料集

```bash
# 從官方來源下載並解壓縮（請依官方指引操作）
# 原始結構：VisA_highres/<category>/{Data/Images/Anomaly,Data/Images/Normal,Annotations/...}
```

#### 步驟 2：使用 spot-diff-main 的 prepare_data 轉換格式

```bash
# clone spot-diff-main（僅用其資料轉換工具）
git clone https://github.com/<spot-diff-repo>/spot-diff-main.git
cd spot-diff-main

# 執行格式轉換（將原始 VisA 轉為 MVTec-AD 相容格式）
python prepare_data.py \
    --data-root /path/to/VisA_highres \
    --output-dir /path/to/VisA_converted \
    --dataset visa
```

轉換完成後的目錄結構（與 MVTec-AD 完全相同）：

```
VisA_converted/
├── candle/
│   ├── train/
│   │   └── good/             ← 正常訓練圖片
│   ├── test/
│   │   ├── good/             ← 正常品測試圖
│   │   └── melted/           ← 缺陷類型（每種一個子目錄）
│   └── ground_truth/
│       └── melted/
│           └── 000_mask.png
├── capsules/
├── cashew/
├── chewinggum/
├── fryum/
├── macaroni1/
├── macaroni2/
├── pcb1/
├── pcb2/
├── pcb3/
├── pcb4/
└── pipe_fryum/
```

---

## 訓練

### 零樣本跨資料集評測協議

SSVP 的標準零樣本評測方式：

| 訓練集 | 測試集 | 說明 |
|--------|--------|------|
| **VisA** | **MVTec-AD** | 兩個資料集類別不重疊，確保零樣本條件 |
| **MVTec-AD** | **VisA** | 反向評測 |

> 論文中採用「在 VisA 上訓練、在 MVTec-AD 上評測（或反之）」的協議，確保測試類別模型從未見過。

---

### 在 VisA 上訓練（推薦工作流程）

每次針對一個 VisA 類別訓練，訓練完成後模型可用於 MVTec-AD 的任意類別推論。

```bash
# 訓練單一 VisA 類別（以 pcb1 為例）
python ssvp/train.py \
    --data-root  /path/to/VisA_converted \
    --category   pcb1 \
    --clip-ckpt  clip-weight/ \
    --dino-ckpt  dino-weight/dinov3_vitb16_pretrain_lvd1689m.pth \
    --output-dir checkpoints/visa/
```

若想訓練全部 12 個 VisA 類別（建議在 bash 中執行）：

```bash
VISA_CATEGORIES="candle capsules cashew chewinggum fryum macaroni1 macaroni2 pcb1 pcb2 pcb3 pcb4 pipe_fryum"

for category in $VISA_CATEGORIES; do
    echo "=== Training: $category ==="
    python ssvp/train.py \
        --data-root  /path/to/VisA_converted \
        --category   $category \
        --clip-ckpt  clip-weight/ \
        --dino-ckpt  dino-weight/dinov3_vitb16_pretrain_lvd1689m.pth \
        --output-dir checkpoints/visa/$category \
        --epochs     10 \
        --batch-size 16
done
```

訓練完成後的 checkpoint 路徑：

```
checkpoints/
└── visa/
    ├── pcb1/
    │   ├── ssvp_pcb1_best.pt   ← Pixel-AUROC 最高的 checkpoint
    │   └── ssvp_pcb1_last.pt   ← 最後一個 epoch
    ├── candle/
    └── ...
```

---

### 在 MVTec-AD 上訓練（反向協議）

```bash
MVTEC_CATEGORIES="bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper"

for category in $MVTEC_CATEGORIES; do
    python ssvp/train.py \
        --data-root  /path/to/mvtec \
        --category   $category \
        --clip-ckpt  clip-weight/ \
        --dino-ckpt  dino-weight/dinov3_vitb16_pretrain_lvd1689m.pth \
        --output-dir checkpoints/mvtec/$category
done
```

---

### 完整訓練參數說明

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--data-root` | — | 資料集根目錄（包含各類別子目錄） |
| `--category` | — | 類別名稱（對應子目錄名稱） |
| `--image-size` | `512` | 輸入解析度，必須是 16 的倍數（DINOv3 patch size=16） |
| `--epochs` | `10` | 訓練週期數 |
| `--batch-size` | `16` | 批次大小 |
| `--lr` | `4e-5` | AdamW 學習率 |
| `--weight-decay` | `1e-4` | AdamW 權重衰減 |
| `--tau` | `0.07` | 對齊損失的溫度參數 τ |
| `--margin` | `0.30` | 餘弦邊距閾值 |
| `--lambda-pixel` | `0.50` | 像素層級 Focal-BCE 損失權重 |
| `--lambda-kl` | `1e-4` | VAE KL 散度損失權重 |
| `--lambda-margin` | `0.10` | 邊距損失權重 |
| `--clip-model` | `ViT-B/16` | CLIP 變體（用於推斷特徵維度） |
| `--dino-model` | `dinov3_vitb16` | DINOv3 hub 模型名稱 |
| `--clip-ckpt` | `None` | CLIP 本地路徑（`.pt` 檔案 或 HuggingFace 資料夾） |
| `--dino-ckpt` | `None` | DINOv3 本地 `.pth` 路徑 |
| `--output-dir` | `checkpoints/` | Checkpoint 儲存路徑 |

### 損失函數

```
L_total = L_align + λ_pixel · L_pixel + λ_kl · L_kl + λ_margin · L_margin
```

| 損失項 | 說明 |
|--------|------|
| `L_align` | 溫度 τ 縮放的圖像層級 [正常, 異常] logit 交叉熵 |
| `L_pixel` | 預測異常圖與 Perlin 合成 GT 遮罩之間的 Focal-BCE |
| `L_kl` | VCPG 的 VAE KL 散度（防止表示崩塌） |
| `L_margin` | 正常/異常餘弦相似度違反邊距約束時的懲罰項 |

> **Checkpoint 說明**：每個 checkpoint 只儲存 HSVS + VCPG + VTAM 的可訓練權重（不含凍結的 CLIP 與 DINOv3），因此體積較小。`_best.pt` 為驗證集 Pixel-AUROC 最高時自動儲存的版本。

---

## 推論

推論時使用在**來源資料集**上訓練好的 checkpoint，對**目標資料集**的圖片進行零樣本異常檢測。

### 命令列推論（最常用）

```bash
# 基本用法：推論單張圖片
python ssvp/inference.py \
    --images     /path/to/image.jpg \
    --class-name bottle \
    --checkpoint checkpoints/visa/pcb1/ssvp_pcb1_best.pt \
    --clip-ckpt  clip-weight/ \
    --dino-ckpt  dino-weight/dinov3_vitb16_pretrain_lvd1689m.pth \
    --output-dir results/
```

```bash
# 批次推論：一個 MVTec 類別的所有測試圖片
python ssvp/inference.py \
    --images     /path/to/mvtec/bottle/test/**/*.png \
    --class-name bottle \
    --checkpoint checkpoints/visa/pcb1/ssvp_pcb1_best.pt \
    --clip-ckpt  clip-weight/ \
    --dino-ckpt  dino-weight/dinov3_vitb16_pretrain_lvd1689m.pth \
    --threshold  0.5 \
    --output-dir results/bottle/
```

```bash
# 跨資料集完整評測：VisA 訓練 → MVTec 所有類別推論
MVTEC_CATEGORIES="bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper"
CKPT="checkpoints/visa/pcb1/ssvp_pcb1_best.pt"

for category in $MVTEC_CATEGORIES; do
    python ssvp/inference.py \
        --images     /path/to/mvtec/$category/test/**/*.png \
        --class-name $category \
        --checkpoint $CKPT \
        --clip-ckpt  clip-weight/ \
        --dino-ckpt  dino-weight/dinov3_vitb16_pretrain_lvd1689m.pth \
        --output-dir results/mvtec/$category/
done
```

**推論輸出：** 每張圖片在 `--output-dir` 產生一個 `<檔名>_anomaly_map.png` 熱力圖，終端機同時顯示異常分數與 NORMAL/ANOMALY 判斷。

---

### Python API 推論

```python
import torch
from PIL import Image
from ssvp.inference import build_model, build_transforms, infer_image
from ssvp.utils import apply_colormap

device = torch.device("cuda")

# ── 1. 載入模型（只需初始化一次）──────────────────────────────────
model = build_model(
    clip_name="ViT-B/16",
    dino_name="dinov3_vitb16",
    checkpoint="checkpoints/visa/pcb1/ssvp_pcb1_best.pt",
    device=device,
    clip_ckpt="clip-weight/",          # HuggingFace 資料夾
    dino_ckpt="dino-weight/dinov3_vitb16_pretrain_lvd1689m.pth",
)
clip_tf, dino_tf = build_transforms(image_size=512)

# ── 2. 推論單張圖片 ───────────────────────────────────────────────
pil_img = Image.open("test.jpg").convert("RGB")
result = infer_image(
    model, pil_img,
    class_name="bottle",          # 改為對應 MVTec 類別名稱
    clip_tf=clip_tf,
    dino_tf=dino_tf,
    device=device,
    image_size=512,
)

print(f"異常分數：{result['score']:.4f}")
# score > 0.5 → 異常；score < 0.5 → 正常（閾值可調）

# ── 3. 儲存熱力圖 ─────────────────────────────────────────────────
# result["anomaly_map"] → torch.Tensor (512, 512)，值越高代表越異常
colored = apply_colormap(result["anomaly_map"], colormap="jet")  # (H, W, 3) uint8
Image.fromarray(colored).save("heatmap.png")
```

---

### 推論輸出說明

| 輸出 Key | 型別 | 說明 |
|----------|------|------|
| `score` | `float` | 圖像層級異常分數，越高代表越異常 |
| `anomaly_map` | `Tensor (H, W)` | 像素層級異常熱力圖，值域 [0, 1] |

> **閾值選擇建議：** `--threshold 0.5` 為預設值。若需最佳化 F1-score，應在驗證集上根據 ROC 曲線選擇使 F1 最大的最佳閾值。

---

## 完整快速開始流程

```
【第一步】準備模型權重
  clip-weight/  ← HuggingFace clip-vit-base-patch16 所有檔案
  dino-weight/dinov3_vitb16_pretrain_lvd1689m.pth

【第二步】準備資料集
  VisA_converted/  ← spot-diff-main prepare_data 轉換後的 VisA

【第三步】訓練
  python ssvp/train.py
      --data-root VisA_converted/
      --category  pcb1
      --clip-ckpt clip-weight/
      --dino-ckpt dino-weight/dinov3_vitb16_pretrain_lvd1689m.pth
      --output-dir checkpoints/

【第四步】推論（在 MVTec-AD 上零樣本評測）
  python ssvp/inference.py
      --images     mvtec/bottle/test/**/*.png
      --class-name bottle
      --checkpoint checkpoints/ssvp_pcb1_best.pt
      --clip-ckpt  clip-weight/
      --dino-ckpt  dino-weight/dinov3_vitb16_pretrain_lvd1689m.pth
      --output-dir results/
```

---

## 模型架構

```
輸入圖片 (B, 3, 512, 512)
        │
        ├──[凍結]──► CLIP ViT-B/16 ──► patch tokens  (B, 1024, 512)
        │             patch_size=16
        │             pos_embed 自動雙線性插值至 512×512
        │
        └──[凍結]──► DINOv3 ViT-B/16 ──► 多尺度 patch tokens  3×(B, 1024, 768)
                      patch_size=16, 4 個 register tokens（自動去除）
                                          │
                               ┌──────────▼──────────┐
                               │        HSVS          │
                               │  3 個 ATF Block      │
                               │  雙路徑跨模態注意力   │
                               └──────────┬──────────┘
                                          │ 增強視覺特徵 (B, 1024, 512)
                               ┌──────────▼──────────┐
                               │        VCPG          │
                               │  VAE 編碼器           │
                               │  跨模態注意力         │
                               └──────┬───────┬───────┘
                           條件文字    │       │ KL 損失
                                      │
                               ┌──────▼──────────────┐
                               │        VTAM          │
                               │  MoE 軟門控（4 expert）│
                               │  局部-全局融合        │
                               └──────────┬──────────┘
                                          │
                        異常分數 (B,)  +  異常圖 (B, 512, 512)
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
- [DINOv3 — facebookresearch](https://github.com/facebookresearch/dinov3)
- [AnomalyCLIP](https://github.com/zqhang/AnomalyCLIP) 與 [VCP-CLIP](https://github.com/xiaozhen228/VCP-CLIP) 的方法論參考
- [MVTec AD 資料集](https://www.mvtec.com/company/research/datasets/mvtec-ad)
- [VisA 資料集](https://github.com/amazon-science/spot-diff)

---

> 英文版 README 請參閱 [README.md](./README.md)
