# FPGA README

本資料夾主要放置 AOC Final Project 在 FPGA 上驗證需要的檔案，包含 FPGA bitstream、hardware handoff file、Python Notebook 測試程式、RTL code 與 測資。

整體分成兩個部分：

1. **Systolic Array Unit Test**：單獨驗證 systolic array IP 是否可以在 FPGA 上正確運算，直接由 python code 寫 data 進 BRAM。
2. **Full Design**： ViT Accelerator one block 設計整合 FPGA 上的 DDR 來模擬 RTL 與 DRAM 溝通驗證。

---

# 1. 通俗介紹：怎麼使用 FPGA

> [!IMPORTANT]
> **只要 FPGA 成功開機、電腦成功連線到 Jupyter Notebook，就可以直接按 `Run` 執行 Python code。**
>
> 因為本資料夾已經放入硬體需要的：
>
> - `design_1_wrapper.bit`
> - `design_1_wrapper.hwh`
>
> 所以使用者只要進入 Jupyter Notebook，打開對應的 `.ipynb`，就能透過 Python 和 FPGA 上的硬體互動。
>
> 以下步驟主要是通俗介紹我們實際在 FPGA 上部署與測試的流程。

---

## 1.1 FPGA 開機與連線流程

### Step 1：將開機模式調整至 SD

將 FPGA 板子的 Boot Mode 調整到 **SD** 的位置，代表系統會從 micro SD 卡開機。

---

### Step 2：調整電源供應模式

依照供電方式調整 jumper：

- 使用電源供應器供電：接 **REG**
- 使用 USB 供電：接 **USB**

---

### Step 3：插入 micro SD

將已經燒好 PYNQ image 的 micro SD 卡插入 FPGA。

---

### Step 4：接上電源

接上 FPGA 電源，但先不要急著操作。

---

### Step 5：插上乙太網路線

將乙太網路線一端接 FPGA，另一端接筆電。

```text
FPGA  <------ Ethernet Cable ------>  Laptop
```

---

### Step 6：開機

打開 FPGA 電源後，先等待板子完全開機。

開機時板子的 RGB LED 會先閃彩燈，等到四顆 LED 全亮紅燈或綠燈時，代表板子已經完全開機完成。

---

## 1.2 Windows 網路設定

在 Windows 上依序開啟：

```text
網路和共用中心
  -> 變更介面卡設定
  -> 乙太網路
  -> 內容
  -> IPv4
  -> 內容
```

將 IP 設定為：

```text
IP Address：192.168.2.1
Subnet Mask：255.255.255.0
```

---

## 1.3 確認是否成功連線 FPGA

在檔案總管輸入：

```text
\\192.168.2.99\xilinx
```

帳號與密碼皆為：

```text
Username：xilinx
Password：xilinx
```

如果可以成功連進去，並看到：

```text
\xilinx\jupyter_notebooks
```

就代表 FPGA 已經成功開機，而且筆電也成功連線到 FPGA。


![FPGA network setting](images/image1.jpg)

![FPGA Windows share connection](images/image2.jpg)

---

## 1.4 從 Vivado XSA 取出 bitstream 與 hwh

Vivado 完成 bitstream 產生後，可以 export 出 `.xsa` 檔案。

使用以下任一壓縮軟體開啟 `.xsa`：

- 7-Zip
- WinRAR

從 `.xsa` 中取出：

```text
design.bit
design.hwh
```

或是 Vivado 產生的名稱可能會是：

```text
design_1_wrapper.bit
design_1_wrapper.hwh
```

接著放到 FPGA 的 Jupyter Notebook 資料夾中，例如：

```text
\\192.168.2.99\xilinx\jupyter_notebooks\design
```

資料夾內至少需要有：

```text
design/
├── design_1_wrapper.bit
├── design_1_wrapper.hwh
└── systolic.ipynb
```

---

## 1.5 撰寫並執行 ipynb

Python Notebook 會負責和 FPGA 上的硬體互動，以 systolic 舉例：

- 載入 bitstream
- 透過 `.hwh` 找到硬體 IP
- 使用 MMIO 存取 AXI-Lite register
- 將 activation、weight、bias 寫入 FPGA BRAM
- 啟動 Systolic Array
- 讀回 output
- 和 golden result 比對

撰寫好的 `.ipynb` 也放在同一個 design 資料夾內。

---

## 1.6 開啟 Jupyter Notebook

打開瀏覽器，輸入：

```text
192.168.2.99:9090
```

預設密碼：

```text
xilinx
```

登入後進入自己建立的 design 資料夾，打開 `.ipynb`，直接按 `Run` 或 `Run All` 即可執行 FPGA 測試。


![Jupyter home](images/image3.jpg)

![Jupyter design folder](images/image4.jpg)

![Run notebook](images/image5.jpg)

---

# 2. Systolic Array Unit Test

本章節介紹單獨在 FPGA 上驗證 Systolic Array 的 unit test。

這個測試的目標不是驗證完整 ViT accelerator，而是先確認 systolic array IP 在 FPGA 上可以正確做到：

```text
Activation x Weight + Bias = Output
```

Python Notebook 會把測試資料寫進 FPGA BRAM，啟動 Systolic Array，最後讀回 256 筆 output，並和 `golden.hex` 比對。

---

## 2.1 systolic_RTL 資料夾架構

```text
systolic_RTL/
├── ActivationMem.v
├── BiasMem.v
├── InputLoadFSM.v
├── OutputCaptureFSM.v
├── OutputMem.v
├── Systolic.v
├── SystolicAxiLiteWrapper.v
├── SystolicSystemCore.v
├── WeightMem.v
└── WeightPingPongController.v
```

## 2.2 RTL Module 說明

| 檔案 | 功能說明 |
| --- | --- |
| `Systolic.v` | Systolic Array 核心運算模組，負責讀取 activation、weight、bias，執行 tile-based GEMM，並輸出 opsum。 |
| `SystolicSystemCore.v` | 系統整合核心，將 Systolic core、input loader、BRAM、weight ping-pong、output capture 串接在一起。 |
| `SystolicAxiLiteWrapper.v` | AXI-Lite wrapper，讓 Python 可以透過 MMIO 寫 register、載入資料、啟動運算、讀回狀態與結果。 |
| `ActivationMem.v` | Activation BRAM template，提供 activation 的 32-bit word 讀寫介面。 |
| `WeightMem.v` | Weight BRAM template，提供 weight 的 32-bit word 讀寫介面。 |
| `BiasMem.v` | Bias BRAM template，提供 bias 的 32-bit word 讀寫介面。 |
| `InputLoadFSM.v` | Host/Python 端資料載入 FSM，透過 AXI-Lite 寫入 activation、weight、bias。 |
| `WeightPingPongController.v` | Weight ping-pong buffer 控制器，讓 weight 可以使用 double buffering，減少等待時間。 |
| `OutputCaptureFSM.v` | 擷取 Systolic Array 輸出的 256 筆 opsum，並依序寫入 output memory。 |
| `OutputMem.v` | Output BRAM template，讓 Python 可以讀回硬體計算結果。 |

---

## 2.3 與原本 Systolic Array 的差異

原本的 DLA 資料夾下 Systolic Array README 中，主要模組只有：

```text
src/
├── Act_fifo.v
├── Opsum_acc.v
├── PE_pack.v
└── Systolic.v
```

原本的設計比較像是 RTL simulation 使用的 systolic core，testbench 直接模擬 BRAM 行為，並且一次給 128-bit data。

這次 FPGA 版本為了能真的放到 PYNQ / FPGA 上執行，因此做了以下修改。

---

### 修改 1：BRAM data width 從 128-bit 改為 32-bit

原本 testbench 模擬的 BRAM 是：

```text
128-bit / word
```

也就是一次讀取一整列 128-bit 的資料。

但在這次 FPGA 實作中，為了符合目前 BRAM / AXI-Lite 搬資料的方式，改成：

```text
32-bit / word
```

因此 RTL 內部會連續讀取四筆 32-bit word，再組回原本 Systolic Array 需要的 128-bit row。

簡單來說：

```text
原本：
1 cycle 讀 128-bit

FPGA 版本：
連續讀 4 個 32-bit word
再組成 1 個 128-bit row
```

這樣雖然資料載入需要更多 cycle，但比較符合 FPGA BRAM 與 Python MMIO 寫入資料的方式。

---

### 修改 2：新增 BRAM template

原本 simulation 中，BRAM 是在 testbench 內用 reg array 模擬。

FPGA 版本新增了實際可合成的 memory module：

```text
ActivationMem.v
WeightMem.v
BiasMem.v
OutputMem.v
```

這些模組在 simulation 時可以用 reg array，在 synthesis 時則會對應到 FPGA 上的 BRAM template。

---

### 修改 3：新增 InputLoadFSM

原本測試資料大多由 testbench 直接餵給硬體。

FPGA 版本需要讓 Python 透過 AXI-Lite 把資料寫進硬體，因此新增：

```text
InputLoadFSM.v
```

它使用簡單的 register-like 介面：

```text
CTRL  ：開始載入，以及選擇 target
BASE  ：寫入起始 address
COUNT ：要寫入幾筆 32-bit word
DATA  ：實際寫入的 data word
```

target 目前包含：

```text
0：activation
1：bias
2：weight
```

---

### 修改 4：新增 AXI-Lite Wrapper

為了讓 Python Notebook 可以控制 FPGA 硬體，新增：

```text
SystolicAxiLiteWrapper.v
```

此模組負責把 AXI-Lite transaction 轉成內部控制訊號，例如：

- 設定 `k_tile_cnt`
- 設定 activation base address
- 設定 weight base address
- 設定 bias base address
- 啟動 systolic array
- 讀取 status
- 讀取 output
- 讀取 profiling counter

---

### 修改 5：新增 Output Capture 與 Output Memory

Systolic Array 會連續輸出 256 筆 opsum。

FPGA 版本新增：

```text
OutputCaptureFSM.v
OutputMem.v
```

流程如下：

```text
Systolic opsum output
        ↓
OutputCaptureFSM
        ↓
OutputMem
        ↓
Python MMIO read back
```

Python 最後會從 output memory 讀回 256 筆資料，並與 `golden.hex` 比對。

---

### 修改 6：新增 Weight Ping-Pong Controller

為了減少 weight loading 造成的等待時間，FPGA 版本新增：

```text
WeightPingPongController.v
```

概念是使用兩個 weight buffer：

```text
Buffer A：目前 systolic 正在讀取
Buffer B：下一個 tile 的 weight 正在載入
```

當下一個 tile 載入完成後，兩個 buffer 交換角色。

這樣可以讓 loading 和 compute 有機會重疊，減少 systolic array 等 weight 的時間。

---

### 修改 7：新增硬體計數器

FPGA 版本也加入一些硬體 profiling counter，方便觀察效能，例如：

- total cycles
- compute busy cycles
- weight load cycles
- overlap cycles
- weight stall cycles
- activation BRAM reads
- weight BRAM reads
- bias BRAM reads
- output word writes

這些 counter 是從 RTL 內部量測，不是 Python 估出來的。

---

## 2.4 FPGA_AOC_systolic 資料夾架構

```text
FPGA_AOC_systolic/
├── hex/
├── design_1_wrapper.bit
├── design_1_wrapper.hwh
├── sds_trace_data.dat
└── systolic.ipynb
```
---

## 2.5 FPGA_AOC_systolic 檔案說明

| 檔案 / 資料夾 | 說明 |
| --- | --- |
| `hex/` | Unit test 測資，沿用原本 Systolic Array 測試那邊提供的資料，包含 activation、weight、bias、golden output。 |
| `design_1_wrapper.bit` | Vivado 產生的 FPGA bitstream，Python Notebook 會透過 PYNQ Overlay 將它載入 FPGA。 |
| `design_1_wrapper.hwh` | Hardware handoff / hardware description file，提供 PYNQ 解析 IP、register map、address map 使用。 |
| `systolic.ipynb` | FPGA unit test 的 Python Notebook，負責載入 bitstream、寫入測資、啟動硬體、讀回 output 並比對 golden。 |
| `sds_trace_data.dat` | PYNQ / Vitis 相關 trace data，主要作為 profiling 或工具產生的輔助檔案。 |

---

## 2.6 hex 測資說明

`hex/` 內存放測資。

每個 case 會包含：

```text
act.hex
weight.hex
bias.hex
golden.hex
```

其中：

- `act.hex`：activation input
- `weight.hex`：weight input
- `bias.hex`：bias input
- `golden.hex`：正確答案，用來和 FPGA output 比對

Python Notebook 會依照 case 設定讀取這些檔案，並透過 AXI-Lite / MMIO 寫入硬體。

---

## 2.7 systolic.ipynb 執行流程

`systolic.ipynb` 的主要流程如下：

```text
1. 載入 bitstream
2. 透過 .hwh 找到 SystolicAxiLiteWrapper IP
3. 建立 MMIO 物件
4. 讀取 hex/case*/act.hex、weight.hex、bias.hex、golden.hex
5. 將 activation 寫入 ActivationMem
6. 將 bias 寫入 BiasMem
7. 將 weight tile 寫入 WeightMem / Ping-Pong Buffer
8. 設定 k_tile_cnt、base address
9. 啟動 Systolic Array
10. 等待硬體完成
11. 讀回 256 筆 output
12. 和 golden.hex 比對
13. 印出 PASS / FAIL 與硬體 counter
```

通過時會看到類似：

```text
case1: PASS
case2: PASS
case3: PASS
case4: PASS
ALL PASS
```

---

# 3. Full Design

本章節介紹 ViT full design 在 FPGA 上的整合與驗證方式。  
Full design 的目標不是單獨驗證某一個 module，而是將 Patch Embedding、Transformer block、Systolic Array、PPU、RMSNorm、Softmax、AXI-Lite control、AXI master DDR 搬移整合成一個可以在 PYNQ-Z2 上執行的 one-block ViT accelerator。

目前 GitHub 中與 Full Design 相關的資料夾分成兩個部分：

```text
FPGA/
├── VIT_fulldesign_fpga/      # 放到 PYNQ / Jupyter Notebook 上執行的 FPGA package
└── vit_fulldesign_rtl/       # Full design RTL source code
```

---

## 3.1 Folder Structure

### FPGA/VIT_fulldesign_fpga/

此資料夾是實際放到 PYNQ board 上執行的 package，包含 bitstream、hwh、notebook、測資與 Vivado reports。

```text
FPGA/VIT_fulldesign_fpga/
├── golden_gen
├── reports
├── vit.ipynb
├── vit_fulldesign.bit
└── vit_fulldesign.hwh
```

| 檔案 / 資料夾 | 說明 |
| --- | --- |
| `vit.ipynb` | PYNQ Jupyter Notebook，負責 load bitstream、配置 DDR buffer、控制 FPGA IP、讀回 counter 與 output。 |
| `vit_fulldesign.bit` | Vivado 產生的 FPGA bitstream。 |
| `vit_fulldesign.hwh` | PYNQ 用來解析 IP name、register map、address map 的 hardware handoff file。 |
| `golden_gen/` | 產生與保存 one-block real-model 測資與 golden output。 |
| `reports/` | Vivado implementation 產生的 timing、utilization、power report。 |

---

### FPGA/vit_fulldesign_rtl/

此資料夾保存 full design 的 RTL source code。

```text
FPGA/vit_fulldesign_rtl/
├── PPU
├── PPU_RMSNorm_Fusion
├── Streaming_RMSNorm_Unit
├── scripts
├── softmax_FPGA_package/softmax_FPGA_package
├── systolic_array/src
├── Global_Controller_FSM.sv
├── Patch_Embedding_Systolic_Top.sv
├── ViT_Accelerator_Top.sv
├── ViT_DDR_AxiLite_Wrapper.sv
├── ViT_Image_Accelerator_Top.sv
├── ViT_InputLoadFSM.sv
├── ViT_System_Core.sv
└── exp_lut_10bit_Q1_15_range12.hex
```

| 檔案 / 資料夾 | 功能 |
| --- | --- |
| `ViT_DDR_AxiLite_Wrapper.sv` | Full design 最外層 wrapper，提供 AXI-Lite control register 與 AXI master DDR load/store。 |
| `ViT_Image_Accelerator_Top.sv` | Image-level top，整合 image input、patch embedding 與 transformer block。 |
| `ViT_System_Core.sv` | 系統核心，負責串接 input loader、patch embedding、ViT accelerator 與 DDR/page request。 |
| `ViT_Accelerator_Top.sv` | Transformer block datapath，包含 RMSNorm、QKV、Attention、MLP、Residual Add 等 phase。 |
| `Patch_Embedding_Systolic_Top.sv` | 使用 systolic array 執行 patch embedding。 |
| `Global_Controller_FSM.sv` | 控制 accelerator phase / stage 的全域 FSM。 |
| `ViT_InputLoadFSM.sv` | 將 DDR 或 host 載入的資料寫進指定 on-chip buffer。 |
| `systolic_array/src` | Systolic Array 相關 RTL。 |
| `PPU` | Requant、GELU、Residual Add 等 post-processing module。 |
| `PPU_RMSNorm_Fusion` | RMS statistic、Gamma buffer、Token stat buffer 等 RMSNorm / PPU fusion 相關 RTL。 |
| `Streaming_RMSNorm_Unit` | Streaming RMSNorm unit 與 RowPacker。 |
| `softmax_FPGA_package/softmax_FPGA_package` | Softmax RTL、exponential LUT、testbench package。 |
| `scripts` | Vivado packaging / report script。 |
| `exp_lut_10bit_Q1_15_range12.hex` | Softmax exponential LUT。 |

---

## 3.2 Full ViT Accelerator 架構

Full design 採用一個 shared systolic array 作為主要 matrix multiplication engine。  
Patch Embedding、QKV Projection、QK^T、A × V、Output Projection、FC1、FC2 都共用同一個 systolic array，而不是為每個 operation 各自建立一份 PE array。

整體架構可以分成五個部分：

```text
PS / Python Notebook
        ↓ AXI-Lite control
ViT_DDR_AxiLite_Wrapper
        ↓
ViT_System_Core
        ↓
Patch Embedding + Transformer Block
        ↓
AXI master DDR load/store
        ↓
DDR memory
```

主要硬體模組：

```text
ViT_DDR_AxiLite_Wrapper
    ├── AXI-Lite register interface
    ├── AXI master DDR loader / storer
    └── performance counters

ViT_System_Core
    ├── InputLoadFSM
    ├── Patch_Embedding_Systolic_Top
    └── ViT_Accelerator_Top

ViT_Accelerator_Top
    ├── RMSNorm1 / RMSNorm2
    ├── QKV Projection
    ├── QK^T + Softmax + A×V
    ├── Output Projection
    ├── Residual Add
    ├── FC1 + GELU
    └── FC2
```

由於 PYNQ-Z2 的 FPGA resource 有限，full design 沒有把完整 12-layer ViT 全部展開成硬體，而是先實作 one-block accelerator，並用 Python 控制測試 one block 的 correctness 與 performance。

---

## 3.3 Dataflow

Full design 的 dataflow 如下：

```text
Input image / position / cls token
        ↓
Patch Embedding
        ↓
X buffer
        ↓
RMSNorm1
        ↓
QKV Projection
        ↓
QK^T
        ↓
Softmax
        ↓
A × V
        ↓
Output Projection
        ↓
Residual Add 1 / X_mid
        ↓
RMSNorm2
        ↓
FC1 + GELU
        ↓
FC2
        ↓
Residual Add 2 / X_out
        ↓
DDR store output
        ↓
Python compare with golden
```

設計目標是盡量讓中間 activation 保留在 on-chip BRAM 中，避免每一層都回 DDR。  
例如：

- `X` 保存在 activation buffer 中，給 RMSNorm1 和 residual add 使用。
- `Score` 和 `A` 共用同一塊 intermediate BRAM。
- `V` 和 `X_mid` 因為 lifetime 不重疊，所以可以共用 buffer。
- `GELU_out` 因為資料量太大，改成 page cache + DDR streaming。

這樣可以減少 DRAM access，但同時因為 PYNQ-Z2 BRAM 有限，所以部分資料仍需要透過 DDR page 方式暫存。

---

## 3.4 Layer Scheduler

Full design 內部由 FSM 控制不同 computation phase。  
目前 one-block 的主要 phase 可以整理成：

```text
Patch Embedding
RMSNorm1
QKV / QK^T / Softmax / A×V
Output Projection
Residual Add 1 / X_mid
RMSNorm2
FC1 + GELU
FC2
Residual Add 2 / X_out
```

Scheduler 的工作包含：

1. 決定目前要執行哪一個 phase。
2. 發出 systolic array 的 enable / tile setting。
3. 決定 activation、weight、bias、residual 要從哪個 buffer 讀。
4. 決定 PPU mode，例如 requant、GELU、residual add。
5. 控制 RMSNorm 的 input stream、gamma address、token statistic address。
6. 控制 Softmax 讀取 score row 並寫回 attention probability。
7. 在需要外部資料時，向 Python / DDR wrapper 發出 tile request 或 page request。
8. 在每個 stage 完成時更新 status / counter，供 Python debug 或 profiling 使用。

目前硬體 scheduler 是針對 one transformer block 設計。  
完整 12 blocks 若要執行，可以由 Python replay 同一顆 one-block accelerator 多次，但 RTL 內部目前不是完整 12-layer automatic scheduler。

---

## 3.5 DMA / Memory Map

Full design 不再只靠 Python 透過 AXI-Lite 一筆一筆寫 BRAM，而是加入 AXI master，讓 RTL 可以主動從 DDR 搬資料。

資料搬移方式：

```text
Python allocate DDR buffer
        ↓
Python 將 hex 測資放入 DDR buffer
        ↓
RTL AXI master 從 DDR load 到 on-chip BRAM / page buffer
        ↓
Accelerator compute
        ↓
RTL AXI master 將 output store 回 DDR
        ↓
Python read back output buffer
```

### AXI-Lite Control

AXI-Lite register 用來做控制與 debug，例如：

```text
start / status
run mode
input target
input count
DDR source address
DDR destination address
DDR word count
page request
tile request
performance counter
debug status
```

Python notebook 透過 `.hwh` 找到 `ViT_DDR_AxiLite_Wrap_0` IP，並使用 MMIO 讀寫這些 register。

### DDR / BRAM Target

Full design 中常見的資料 target 包含：

```text
image input
position embedding
cls token
gamma
patch weight
patch bias
transformer weight
transformer bias
GELU page buffer
X / X_out buffer
```

其中 weight、bias、image、position 等資料會先由 Python 放到 DDR buffer，再由 RTL AXI master load 到對應 on-chip buffer。

---

## 3.6 Systolic Array 與 PPU / RMSNorm / Softmax 的整合方式

### Systolic Array

Systolic Array 是主要 GEMM engine，用於：

```text
Patch Embedding
QKV Projection
QK^T
A × V
Output Projection
FC1
FC2
```

早期設計曾使用 16×16 systolic，但 PYNQ-Z2 上 DSP、LUT、routing 壓力太大，因此 full design 後來改為較小的 shared systolic 架構。  
這會增加 cycle 數，但可以讓設計通過 implementation。

### PPU

PPU 負責 systolic output 後處理，例如：

```text
Requant
GELU
Residual Add
RMS statistic accumulation
```

為了減少 LUT 與 routing 壓力，PPU 不再一次處理完整 tile，而是偏向 streaming valid/ready 方式逐筆處理。

### RMSNorm

RMSNorm 使用 streaming dataflow：

```text
X / X_mid buffer
        ↓
Streaming RMSNorm
        ↓
normalized activation buffer
```

RMSNorm 會讀取：

```text
Token Stat SRAM: inv_rms[t]
Gamma Buffer: gamma[c]
```

並以 token-major 順序輸出 normalized activation。

### Softmax

Softmax 使用 `exp_lut_10bit_Q1_15_range12.hex` 作為 exponential LUT。

Attention flow：

```text
QK^T output score
        ↓
Softmax
        ↓
attention probability A
        ↓
A × V
```

Score 和 attention probability 共用 intermediate buffer，Softmax 後會把 score 覆寫成 A，以節省 BRAM。

---

## 3.7 Full Design Python Control Flow

`vit.ipynb` 是 full design 的主要 FPGA 控制程式。

執行流程：

```text
1. Load vit_fulldesign.bit
2. 透過 vit_fulldesign.hwh 找到 ViT_DDR_AxiLite_Wrap_0
3. 建立 MMIO object
4. 讀取 golden_gen/hex/case_vit_real_model 測資
5. 使用 PYNQ allocate 建立 DDR buffer
6. 將 image / pos / cls / gamma / weight / bias 放入 DDR
7. 設定 AXI-Lite control registers
8. 啟動 accelerator
9. Python 持續服務 RTL 發出的 tile request / page request
10. RTL 透過 AXI master load/store DDR data
11. 等待 one-block 完成
12. store X_out 到 DDR
13. Python read back X_out
14. 和 software golden 比對
15. 讀取 performance counters
```

Notebook 也會印出 progress，例如目前 phase、tile request、DDR 狀態、counter 數值等，方便觀察硬體是否卡在某個 stage。

---

## 3.8 Full Design FPGA Validation Result

Full design 的驗證包含兩個層面：

### Correctness validation

使用 `golden_gen/hex/case_vit_real_model` 中的 real-model 測資，將 FPGA output 與 software golden 比對。

主要 golden 檔案：

```text
x_out.hex
x_out.npy
stage_golden_block0.npz
```

其中：

- `x_out.hex` / `x_out.npy` 用來比對 one-block 最終 output。
- `stage_golden_block0.npz` 可用於 stage-level debug。
- `x_after_patch.npy`、`x_mid.npy` 可用於中間結果檢查。

### Performance validation

Full design RTL 中加入 performance counters。  
Python notebook 會讀取並整理：

```text
total cycles
compute busy cycles
DDR load cycles
DDR store cycles
DDR read / write words
BRAM read / write words
BRAM active cycles
MAC count
DDR transaction count
GELU page wait cycles
GELU page overlap cycles
```

這些 counter 可用來計算：

```text
BRAM bandwidth
DRAM access time
MAC/cycle
DDR bytes / transaction
ping-pong overlap ratio
energy estimate
```

### Vivado implementation reports

`reports/` 中保存 implementation 後的 report：

```text
reports/
├── power_impl_vectorless.rpt
├── timing_impl_vectorless.rpt
└── utilization_impl_hier.rpt
```

其中：

- `utilization_impl_hier.rpt`：查看 LUT / FF / BRAM / DSP 使用量。
- `timing_impl_vectorless.rpt`：確認 timing 是否 meet。
- `power_impl_vectorless.rpt`：估算 on-chip power。

目前 full design 已能完成 Vivado implementation，並可產生 timing、utilization、power report。  
FPGA notebook 也可以讀取 performance counters，用於和理論 profiling 比較。

---

## 3.9 注意事項

1. `vit.ipynb`、`vit_fulldesign.bit`、`vit_fulldesign.hwh` 必須放在同一層資料夾。
2. `golden_gen/hex/case_vit_real_model` 必須存在，否則 notebook 找不到測資。
3. `.bit` 和 `.hwh` 必須來自同一次 Vivado build，否則 PYNQ 可能找不到正確 IP 或 address map。
4. 若修改 RTL，必須重新 package IP、generate bitstream，並更新 `.bit` / `.hwh`。
5. 若 AXI master 沒有正確接到 PS DDR，notebook 會在 DDR load/store 階段 timeout。
6. 目前 full design 主要驗證 one-block accelerator，不是完整 12-block all-hardware scheduler。
---

