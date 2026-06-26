# AOC_final_project
本專案主要包含以下5個子目錄, 各子目錄的README有詳細的Requirement 和 檔案說明
```
.
├── PT_DIR/                     # 把 rms_qat_best.pt 權重檔放這裡
├── model_verification/         # Pytorch model optimize & eval
├── Profiling/
├── DLA/                        # RTL simulation & DLA_model.py
├── FPGA/                       # Full Design FPGA verifivation
└── README.md
```
## PT_DIR/
==請將載下來的權重檔放在這個資料夾下==
```
AOC_final_project
├── PT_DIR
│   ├── README.md
│   └── rms_qat_best.pt
|
```

## [model_verification/](./model_verification/README.md)
收錄軟體端的驗證實驗——也就是在 DLA RTL 開始設計之前，用 Python/PyTorch 驗證模型架構（RMSNorm）、量化方法（PTQ/QAT、Scale Tying、Bit-Width）與 FlashAttention 是否可行、精度代價多少的全部 notebook。

## [Profiling/](./Profiling/README.md)
比較 **Baseline-B FP32 hardware-aware tiled model** 與 **Optimized INT8 RTL-aware model** 在一個 ViT-Small/16 Transformer block 上的理論 profiling 結果。
Profiler 對齊目前 FPGA RTL 設計：shared 8×8 systolic array、INT8 optimized dataflow、BRAM reuse、weight ping-pong、Softmax/RMSNorm/GELU LUT，以及 MLP GELU page streaming。

## [DLA/](./DLA/README.md)
主要包含
- 四個RTL module 的實作及unit test
  - PPU
  - Softmax
  - Streaming_RMSNorm_Unit
  - Systolic 
- 用python實作的`DLA演算法模型`DLA_model.py, 
  對應的是我們預期DLA硬體要執行的演算法, 並用來進行DLA推論與Pytorch 的比較, 評估硬體運算精確度。

## [FPGA/](./FPGA/README.md)
主要放置Final Project 在 FPGA 上驗證需要的檔案，包含
- FPGA bitstream
- Hardware handoff file
- Python Notebook 測試程式
- RTL code
- 測資

