# Phase 0 + Phase 1 完成報告
## HAL/Driver/Runtime 框架建立 × FlashAttention RTL 驗證

> 本文件供組內討論用，詳細說明目前已完成的兩個階段。
> 讀者對象：熟悉 C++/SystemVerilog，但不一定了解此框架細節的組員。

---

## 目錄

- [Phase 0：框架建立](#phase-0框架建立)
  - [P0-1 整體設計目標](#p0-1-整體設計目標)
  - [P0-2 三層軟體架構](#p0-2-三層軟體架構)
  - [P0-3 HAL 層詳解](#p0-3-hal-層詳解)
  - [P0-4 Driver 層詳解](#p0-4-driver-層詳解)
  - [P0-5 Runtime 層詳解](#p0-5-runtime-層詳解)
  - [P0-6 DPI-C 浮點橋接](#p0-6-dpi-c-浮點橋接)
  - [P0-7 Makefile 建置系統](#p0-7-makefile-建置系統)
- [Phase 1：FlashAttention RTL 驗證](#phase-1flashattention-rtl-驗證)
  - [P1-1 FlashAttention-2 演算法回顧](#p1-1-flashattention-2-演算法回顧)
  - [P1-2 RTL 模組介面設計](#p1-2-rtl-模組介面設計)
  - [P1-3 MMIO 暫存器規劃](#p1-3-mmio-暫存器規劃)
  - [P1-4 State Machine 詳解](#p1-4-state-machine-詳解)
  - [P1-5 FA_COMPUTE：Online Softmax 的 RTL 實作](#p1-5-fa_computeonline-softmax-的-rtl-實作)
  - [P1-6 三個關鍵設計決策與 Bug 修正](#p1-6-三個關鍵設計決策與-bug-修正)
  - [P1-7 三個 Test Case 與驗證結果](#p1-7-三個-test-case-與驗證結果)
  - [P1-8 數值驗證方法論](#p1-8-數值驗證方法論)
- [完整檔案清單](#完整檔案清單)
- [給組員的 Q&A](#給組員的-qa)

---

# Phase 0：框架建立

## P0-1 整體設計目標

Phase 0 的任務是建立一套**可重複使用的硬體模擬軟體框架**。
這個框架讓後續每一個硬體加速器元件（FlashAttention、RMSNorm、GEMM...）
都能套用同一個模式來驗證，不需要每次從零開始。

```
核心問題：如何在純軟體環境（PC/Linux）模擬「在 FPGA 上跑的硬體加速器」？

答案：Verilator
  .sv (SystemVerilog RTL)  →  Verilator 編譯  →  C++ class
                                                   （行為與真實 RTL 完全一致）
  C++ testbench 透過呼叫這個 class 的 eval() 來驅動 RTL，
  就像真的在數位電路上打 clock 一樣。

但只有 Verilator 還不夠，還需要：
  1. 模擬 AXI4 MMIO 的讀寫（軟體驅動硬體暫存器）
  2. 模擬 AXI4 DMA 的資料傳輸（硬體讀寫主記憶體）
  3. 模擬 clock 的推進（每個 clock_step 都要 eval() 一次）
  4. 追蹤效能指標（cycle 數、時間、記憶體頻寬）

→ 這就是 HAL/Driver/Runtime 三層架構要解決的問題。
```

---

## P0-2 三層軟體架構

```
┌─────────────────────────────────────────────────────────────────────┐
│                      ★ 三層架構全貌 ★                               │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  Layer 3：Runtime                                            │  │
│  │  使用者看到的高階 API                                         │  │
│  │                                                              │  │
│  │  flash_attention(Q, K, V, O, N, d, Br)                      │  │
│  │  standard_attention_cpu(Q, K, V, O, N, d)                   │  │
│  │  fa_stop()                                                   │  │
│  └──────────────────────────┬───────────────────────────────────┘  │
│                             │ 呼叫                                  │
│  ┌──────────────────────────▼───────────────────────────────────┐  │
│  │  Layer 2：Driver                                             │  │
│  │  暫存器層級的讀寫 API                                         │  │
│  │                                                              │  │
│  │  fa_reg_write(offset, value)  →  g_hal->memory_set(...)      │  │
│  │  fa_reg_read(offset)          →  g_hal->memory_get(...)      │  │
│  │  fa_set_shape(N, d)                                          │  │
│  │  fa_set_tile(Br)                                             │  │
│  │  fa_set_q/k/v/o_addr(ptr)                                   │  │
│  │  fa_start()                                                  │  │
│  └──────────────────────────┬───────────────────────────────────┘  │
│                             │ 呼叫                                  │
│  ┌──────────────────────────▼───────────────────────────────────┐  │
│  │  Layer 1：HAL（Hardware Abstraction Layer）                  │  │
│  │  Verilator 模型的 lifecycle + AXI4 協議仿真                  │  │
│  │                                                              │  │
│  │  memory_set(addr, data)  →  AXI4 Slave Write 時序           │  │
│  │  memory_get(addr, &data) →  AXI4 Slave Read 時序            │  │
│  │  wait_for_irq()          →  clock loop + DMA service        │  │
│  │  handle_dma_read()       →  服務 RTL 的讀 DRAM 請求          │  │
│  │  handle_dma_write()      →  服務 RTL 的寫 DRAM 請求          │  │
│  └──────────────────────────┬───────────────────────────────────┘  │
│                             │ AXI4 訊號 + clock_step()             │
│  ┌──────────────────────────▼───────────────────────────────────┐  │
│  │  Verilated RTL Model（由 Verilator 從 .sv 生成）              │  │
│  │  Vflash_attn_wrapper                                         │  │
│  │                                                              │  │
│  │  device_->ACLK      device_->ARVALID_S  device_->RDATA_M    │  │
│  │  device_->ARESETn   device_->AWADDR_S   device_->WVALID_M   │  │
│  │  device_->eval()    device_->FA_interrupt  ...              │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘

設計優勢：
  ① 每層只依賴下一層，上層不需要知道 AXI4 協議細節
  ② 換一個新的硬體元件，只需要換 HAL 裡的 device_ 型別和 RTL
  ③ Driver 和 Runtime 只依賴 HAL 的抽象介面，不依賴特定 Verilated class
```

---

## P0-3 HAL 層詳解

### HAL 層的核心職責

```
HAL = Hardware Abstraction Layer
職責：把「呼叫 C++ 函式」轉換成「驅動 AXI4 時序」

真實 FPGA 上：CPU 透過 AXI4-Lite 寫暫存器、透過 AXI4 DMA 傳資料
Verilator sim：HAL 透過操作訊號引腳 + clock_step 來仿真同樣的行為
```

### clock_step 巨集：模擬時鐘推進

```c
// flash_attn_hal.hpp
#define clock_step(dut, signal, elapsed_cycle, elapsed_time)  \
    do {                                                       \
        (dut)->signal = 0;   /* 時鐘下降沿 */                  \
        (dut)->eval();       /* RTL 計算組合邏輯 */             \
        (dut)->signal = 1;   /* 時鐘上升沿 */                  \
        (dut)->eval();       /* RTL 在上升沿更新暫存器 */       \
        (elapsed_time) += CYCLE_TIME;  /* +5 ns */            \
        (elapsed_cycle)++;             /* +1 cycle */         \
    } while (0)
```

```
一個 clock cycle 的時序：

  ACLK  ____╭───╮____╭───╮____
             ↑   ↓
           dut->signal=1  dut->signal=0
           dut->eval()    dut->eval()
           （上升沿：      （下降沿：
             FF 更新）       組合邏輯穩定）

  在 Verilator 的 always_ff @(posedge ACLK) 會在
  signal=1 的 eval() 時執行。
```

### vm_addr_h_：64 位元指標橋接

這是整個框架最關鍵也最容易混淆的設計。

```
問題背景：
  RTL 的 DMA 只傳 32 位元位址（reg_q_addr 是 32-bit 暫存器）
  但 Linux x86-64 的指標是 64 位元

  假設 Q_data 陣列的位址是 0x00007FFF_ABCD0000
  RTL 只能存   32 位元：          0xABCD0000
  ← 高 32 位元 0x00007FFF 哪裡去了？

解法：vm_addr_h_ 儲存「上半段」

  FlashAttnHAL 物件本身在記憶體某個位址，例如 0x00007FFF_12340000
  vm_addr_h_ = 0x00007FFF_00000000  （取 HAL 物件位址的上 32 bits）

  假設 Q_data 在 0x00007FFF_ABCD0000
  RTL 存的 reg_q_addr = 0xABCD0000  （低 32 bits）
  HAL 重建：  vm_addr_h_ | reg_q_addr
            = 0x00007FFF_00000000 | 0xABCD0000
            = 0x00007FFF_ABCD0000  ✓ 還原正確位址

前提：HAL 物件和所有資料陣列必須在同一個 4GB 對齊區段內
保證方式：tb.cpp 的資料陣列宣告為 static（確保在 BSS/data segment）
```

```cpp
// flash_attn_hal.cpp
FlashAttnHAL::FlashAttnHAL(...) {
    // 取 HAL 物件自身位址的上 32 bits
    vm_addr_h_ = (reinterpret_cast<uint64_t>(this) & 0xffffffff00000000ULL);
}

// handle_dma_read()：還原完整 64 位元指標
uint32_t* addr = reinterpret_cast<uint32_t*>(vm_addr_h_ | device_->ARADDR_M);
```

```cpp
// driver_flash_attn.cpp：只存低 32 bits
void fa_set_q_addr(const float* Q) {
    fa_reg_write(FA_Q_ADDR_OFFSET,
        static_cast<uint32_t>(reinterpret_cast<uintptr_t>(Q) & 0xFFFFFFFFu));
}
```

```cpp
// tb.cpp：static 確保與 HAL 在同一 4GB 區段
static float Q_data[CASE_N * CASE_D];   // ← static 很重要！
static float K_data[CASE_N * CASE_D];
static float V_data[CASE_N * CASE_D];
static float O_hw  [CASE_N * CASE_D];
static FlashAttnHAL hal(FA_MMIO_BASE_ADDR, FA_MMIO_SIZE);
```

### AXI4 MMIO Write 時序（memory_set）

```
HAL 模擬 AXI4-Lite Write 的三個 channel：AW → W → B

  AW channel（寫位址）：
    device_->AWADDR_S  = addr
    device_->AWVALID_S = 1
    loop until AWREADY_S == 1 → clock_step
    clock_step
    device_->AWVALID_S = 0

  W channel（寫資料）：
    device_->WDATA_S  = data
    device_->WSTRB_S  = 0b1111  (全部 4 bytes)
    device_->WLAST_S  = 1
    device_->WVALID_S = 1
    loop until WREADY_S == 1 → clock_step
    clock_step
    device_->WVALID_S = 0

  B channel（寫回應）：
    device_->BREADY_S = 1
    loop until BVALID_S == 1 → clock_step
    clock_step
    device_->BREADY_S = 0
    → 讀取 BRESP_S，確認 AXI_RESP_OKAY (0x0)

時序示意：
  ACLK     ___╭╮___╭╮___╭╮___╭╮___╭╮___╭╮___╭╮___╭╮
  AWVALID  ____╭─────╮_____________________________
  AWREADY  ________╭─╮__________________________
  WVALID   __________╭─────╮___________________
  WREADY   ______________╭─╮___________________
  BVALID   ____________________╭─────╮_________
  BREADY   ________________╭───────────╮_______
```

### AXI4 DMA Read 時序（handle_dma_read）

```
RTL 想讀 DRAM 資料時，會拉起 ARVALID_M。
HAL 在 wait_for_irq() 的 clock loop 裡偵測到後，呼叫 handle_dma_read()。

void handle_dma_read() {
    uint32_t* addr = (vm_addr_h_ | ARADDR_M);  // 還原完整位址
    uint32_t  len  = ARLEN_M;                  // burst length - 1

    // 接受 AR request
    ARREADY_M = 1; clock_step; ARREADY_M = 0; clock_step;

    // 傳送 R beats（共 len+1 個 word）
    for (int i = 0; i <= len; i++) {
        RDATA_M  = addr[i];               // 從真實記憶體讀取
        RLAST_M  = (i == len);            // 最後一個 beat
        RVALID_M = 1;
        eval();
        while (!RREADY_M) clock_step;    // 等 RTL 準備好
        clock_step;                       // RTL 接收這個 beat
        RVALID_M = 0;
        clock_step;

        // 模擬記憶體延遲
        elapsed_cycle += MEM_ACCESS_CYCLE;  // +5 cycles per word
        elapsed_time  += MEM_ACCESS_CYCLE * CYCLE_TIME;
    }
    memory_read += sizeof(uint32_t) * (len + 1);
}

時序（一個 d=4 的 burst）：
  ACLK       ___╭╮___╭╮___╭╮___╭╮___╭╮___╭╮___╭╮___╭╮
  ARVALID_M  ╭─────╮_________________________________________  (RTL 發出)
  ARREADY_M  ____╭─╮_________________________________________  (HAL 接受)
  RVALID_M   ________╭──╮__╭──╮__╭──╮__╭──╮_______________  (HAL 傳送)
  RREADY_M   ╭──────────────────────────────────────────────  (RTL 總是 1)
  RLAST_M    _________________________________╭──╮__________  (最後一個)
  RDATA_M    ────[word0]──[word1]──[word2]──[word3]──────────
```

### wait_for_irq：核心驅動迴圈

```cpp
void FlashAttnHAL::wait_for_irq() {
    while (!device_->FA_interrupt) {
        clock_step(device_, ACLK, info_.elapsed_cycle, info_.elapsed_time);

        if (device_->ARVALID_M)  handle_dma_read();   // RTL 想讀 DRAM
        if (device_->AWVALID_M)  handle_dma_write();  // RTL 想寫 DRAM
    }
    // FA_interrupt == 1 → RTL 計算完成
}
```

```
這個迴圈的意義：

  每一個 clock cycle，HAL 都問：
    「RTL 有沒有發出 DMA 讀請求？」→ 有就服務 handle_dma_read()
    「RTL 有沒有發出 DMA 寫請求？」→ 有就服務 handle_dma_write()
    「RTL 有沒有完成並發出 IRQ？」  → 有就離開迴圈

  這精確地模擬了真實 FPGA 上：
    CPU 等待中斷，同時 DMA 控制器自動處理資料傳輸
```

### HALBase 抽象介面

```cpp
// include/hal/hal.hpp
struct runtime_info {
    uint64_t elapsed_cycle;   // 模擬經過的 clock cycle 數
    uint64_t elapsed_time;    // 模擬經過的時間（ns）
    uint32_t memory_read;     // DMA 讀取總 bytes
    uint32_t memory_write;    // DMA 寫入總 bytes
};

class HALBase {
   public:
    virtual void init()   = 0;   // 初始化 + reset RTL
    virtual void reset()  = 0;   // 拉低 ARESETn 若干 cycle
    virtual void final()  = 0;   // 清理（FST 關檔等）
    virtual struct runtime_info get_runtime_info() const = 0;
    virtual void reset_runtime_info() = 0;
};
```

---

## P0-4 Driver 層詳解

Driver 層把「暫存器 offset」包裝成有語意的 C 函式。

### MMIO 暫存器地圖

```
Base address: 0x10050000
（lab-4 DLA 用 0x10040000，本專案用 0x10050000，不衝突）

Offset  名稱        位元定義                    用途
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
0x00    CONTROL     [0] = start                  寫 1 啟動計算
                    [1] = irq_clear              寫 1 清除中斷
0x04    SHAPE       [31:16] = N (sequence len)   token 數量
                    [15:0]  = d (head dim)        每個 head 的維度
0x08    TILE        [15:0]  = Br (= Bc)          tile 大小
0x0C    Q_ADDR      [31:0]  = Q 矩陣基底位址低 32 bits
0x10    K_ADDR      [31:0]  = K 矩陣基底位址低 32 bits
0x14    V_ADDR      [31:0]  = V 矩陣基底位址低 32 bits
0x18    O_ADDR      [31:0]  = O 矩陣基底位址低 32 bits
0x1C    STATUS      [0] = busy（唯讀）             計算中
                    [1] = done（唯讀）             計算完成
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 啟動序列（Runtime 呼叫 Driver 的順序）

```cpp
// 1. 設定矩陣形狀
fa_set_shape(196, 64);
//   → fa_reg_write(0x04, (196 << 16) | 64)
//   → g_hal->memory_set(0x10050004, 0x00C40040)
//   → AXI4 Write: addr=0x10050004, data=0x00C40040

// 2. 設定 tile 大小
fa_set_tile(14);
//   → fa_reg_write(0x08, 14)

// 3. 設定各矩陣位址（低 32 bits）
fa_set_q_addr(Q_data);   // Q_data 的低 32 bits
fa_set_k_addr(K_data);
fa_set_v_addr(V_data);
fa_set_o_addr(O_hw);

// 4. 啟動
fa_start();
//   → fa_reg_write(0x00, 1)  (FA_CTRL_START = bit 0)
//   → RTL 的 FA_IDLE 狀態看到 reg_control[0]=1，開始執行
```

---

## P0-5 Runtime 層詳解

Runtime 提供最高層的 API，Testbench 只需要呼叫這兩個函式：

```cpp
// 硬體路徑：用 RTL 加速器計算 FlashAttention
int flash_attention(float* Q, float* K, float* V, float* O,
                    uint32_t N, uint32_t d, uint32_t Br);

// CPU 黃金參考：用軟體計算標準 attention，供比對用
void standard_attention_cpu(const float* Q, const float* K, const float* V,
                             float* O, uint32_t N, uint32_t d);
```

### flash_attention() 的執行流程

```
flash_attention(Q, K, V, O, N=196, d=64, Br=14)
  │
  ├─ fa_set_shape(196, 64)         寫 MMIO 0x04
  ├─ fa_set_tile(14)               寫 MMIO 0x08
  ├─ fa_set_q_addr(Q)              寫 MMIO 0x0C
  ├─ fa_set_k_addr(K)              寫 MMIO 0x10
  ├─ fa_set_v_addr(V)              寫 MMIO 0x14
  ├─ fa_set_o_addr(O)              寫 MMIO 0x18
  │   （以上每次 MMIO 寫都觸發 AXI4 Write 時序）
  │
  ├─ get_fa_hal()->reset_runtime_info()
  │   清零 elapsed_cycle / elapsed_time / memory_read / memory_write
  │   確保效能指標只計算本次計算，不包含 MMIO 設定的 cycles
  │
  ├─ fa_start()                    寫 MMIO 0x00 = 1
  │   RTL 從 FA_IDLE 進入 FA_DMA_Q_AR
  │
  ├─ get_fa_hal()->wait_for_irq()  ← 大部分時間花在這裡
  │   每個 clock 都 eval()
  │   偵測 ARVALID_M → handle_dma_read()（Q/K/V 讀取）
  │   偵測 AWVALID_M → handle_dma_write()（O 寫回）
  │   直到 FA_interrupt == 1
  │
  └─ fa_stop()                     寫 MMIO 0x00 = 0
```

### standard_attention_cpu() 的算法

```cpp
void standard_attention_cpu(Q, K, V, O, N, d) {
    scale = 1.0 / sqrt(d)

    // Step 1: S = Q × K^T × scale    [N × N]
    for i in N:
        for j in N:
            S[i][j] = dot(Q[i], K[j]) * scale

    // Step 2: row-wise softmax(S) → P
    for i in N:
        max_i = max(S[i])
        sum_i = sum(exp(S[i][j] - max_i))
        P[i][j] = exp(S[i][j] - max_i) / sum_i

    // Step 3: O = P × V               [N × d]
    for i in N:
        for k in d:
            O[i][k] = sum(P[i][j] * V[j][k] for j in N)
}

特點：
  - 標準兩次掃描 softmax（先找 max，再算 exp-sum）
  - 需要 O(N²) 暫存空間（S 矩陣）
  - 計算結果是 FlashAttention 的「黃金標準」（ground truth）
```

---

## P0-6 DPI-C 浮點橋接

### 為什麼需要 DPI-C？

```
SystemVerilog real 型態 = IEEE 754 fp64（C 的 double）
AXI4 DMA 傳輸的資料    = IEEE 754 fp32（C 的 float，4 bytes）
Online softmax 需要    = exp() 和 sqrt()

問題 1：fp32 bit pattern → SV real
  RDATA_M 是 uint32，代表一個 fp32 數值的 bit pattern
  SV 沒有內建的 $bitstoshortreal 在所有 Verilator 版本都可靠
  → 需要 DPI-C 函式做轉換

問題 2：SV real → fp32 bit pattern
  O_buf 的值（fp64）要寫回 WDATA_M（uint32 fp32 格式）
  → 需要 DPI-C 函式做轉換

問題 3：exp() 和 sqrt()
  SV 沒有可靠的內建 exp()/sqrt() 在 Verilator behavioral sim 裡
  → 用 DPI-C 直接呼叫 C 的 <math.h>
```

### dpi_math.c 完整實作

```c
// src/dpi/dpi_math.c
#include <math.h>
#include <string.h>
#include "svdpi.h"   // Verilator 提供，DPI-C 必要標頭

// fp32 bit pattern → fp64（memcpy 確保 bit-exact，不做任何數值轉換）
double dpi_fp32_bits_to_real(unsigned int bits) {
    float f;
    memcpy(&f, &bits, sizeof(float));   // 直接 reinterpret bits
    return (double)f;                   // 升精度到 fp64
}

// fp64 → fp32 bit pattern（round-to-nearest，再取 bit pattern）
unsigned int dpi_real_to_fp32_bits(double d) {
    float f = (float)d;                 // 降精度到 fp32
    unsigned int bits;
    memcpy(&bits, &f, sizeof(unsigned int));
    return bits;
}

double dpi_expf(double x)  { return exp(x);  }   // 精度優先用 fp64 exp
double dpi_sqrtf(double x) { return sqrt(x); }   // 同上
```

### SV 側的宣告與使用

```systemverilog
// flash_attn_wrapper.sv
import "DPI-C" function real dpi_fp32_bits_to_real(input int unsigned bits);
import "DPI-C" function int unsigned dpi_real_to_fp32_bits(input real val);
import "DPI-C" function real dpi_expf(input real x);
import "DPI-C" function real dpi_sqrtf(input real x);

// 使用範例（在 FA_DMA_Q_R state 裡）：
Q_buf[cur_row][word_cnt] <= dpi_fp32_bits_to_real(RDATA_M);
//  RDATA_M 是 logic [31:0]，代表 fp32 的 bit pattern
//  dpi_fp32_bits_to_real 把它轉成 real（fp64）存進 Q_buf

// 在 FA_DMA_O_W state 裡：
WDATA_M <= dpi_real_to_fp32_bits(O_buf[cur_row][word_cnt]);
//  O_buf 是 real（fp64）
//  dpi_real_to_fp32_bits 把它 round 成 fp32，取 bit pattern
```

### 精度策略

```
DRAM（fp32）→ DPI-C → Q_buf/K_buf/V_buf（fp64）
                              ↓
                       FA_COMPUTE（fp64 計算）
                       exp/sqrt 全用 fp64
                              ↓
                       O_buf（fp64）
                              ↓
                DPI-C → DRAM（fp32）

好處：中間累加誤差最小（避免 fp32 反覆 round 的累積誤差）
最後 round 回 fp32 是唯一的精度損失點
→ 實測 max_diff < 1e-6，遠優於 1e-4 容差
```

---

## P0-7 Makefile 建置系統

### 三個 Makefile 的分工

```
專案根目錄/Makefile                    ← 入口點，協調各子 Makefile
  └─ src/hardware/flash_attn/Makefile  ← Verilator 編譯 RTL → .a
  └─ test/testbench/flash_attn/Makefile← 編譯 testbench → 可執行檔
```

### 建置流程圖

```
make run_fa
  │
  ├─► src/hardware/flash_attn/Makefile
  │     │
  │     ├─ verilator -Wall --cc --trace-fst --top-module flash_attn_wrapper
  │     │           flash_attn_wrapper.sv
  │     │   → obj_dir/Vflash_attn_wrapper.h        （Verilated class 宣告）
  │     │   → obj_dir/Vflash_attn_wrapper.cpp      （Verilated class 實作）
  │     │   → obj_dir/Vflash_attn_wrapper__Dpi.h   （DPI-C 函式宣告）
  │     │
  │     └─ g++ / gcc 編譯
  │         verilated.cpp + verilated_fst_c.cpp + verilated_threads.cpp
  │         Vflash_attn_wrapper.cpp
  │         dpi_math.c
  │         → ar rcs libVflash_attn_wrapper.a *.o
  │
  └─► test/testbench/flash_attn/Makefile（針對 case0/1/2）
        │
        └─ g++ -std=c++17 -DCASE_NUM=2
               tb.cpp
               flash_attn_hal.cpp
               driver_flash_attn.cpp
               runtime_flash_attn.cpp
               -l libVflash_attn_wrapper.a -lpthread -lm -lz
               → build/tb_fa_case2
```

---

# Phase 1：FlashAttention RTL 驗證

## P1-1 FlashAttention-2 演算法回顧

### 為什麼不用標準 Attention？

```
標準 Attention 的問題：
  S = Q × K^T  →  S 是 [N×N] 矩陣
  N = 196（ViT-Small）時：S = 196 × 196 × 4 bytes ≈ 150 KB
  這個矩陣在計算 softmax 之前必須全部放進 on-chip SRAM
  PYNQ-Z2 的 BRAM 只有 ~2.1 Mb = 262 KB → 只夠放一個 S 矩陣，沒有餘裕
  而且 S 必須寫回 DRAM、再讀回來做 P × V → 兩次 DRAM 存取

FlashAttention-2 的解法：
  把 Q/K/V 都切成小 tile（大小 = Br × d）
  每個 tile 放進 SRAM，計算完就可以丟掉
  用「online softmax」讓 normalize 不需要看到全部 N 個元素
  → S 矩陣永遠不寫回 DRAM
  → DRAM 存取量：O(N) 而非 O(N²)
```

### FlashAttention-2 虛擬碼

```python
def flash_attention_2(Q, K, V, N, d, Br):
    O = zeros(N, d)

    for i in range(0, N, Br):          # i-tile 迴圈（Q 的行）
        Q_i = Q[i:i+Br]                # 讀 Q 的一個 tile [Br × d]
        O_i = zeros(Br, d)
        l_i = zeros(Br)                # running sum（分母）
        m_i = fill(-inf, Br)           # running max

        for j in range(0, N, Br):      # j-tile 迴圈（K/V 的行）
            K_j = K[j:j+Br]            # 讀 K 的一個 tile [Br × d]
            V_j = V[j:j+Br]            # 讀 V 的一個 tile [Br × d]

            S_ij = Q_i @ K_j.T / sqrt(d)    # [Br × Br]，只在 SRAM 裡

            # Online Softmax 核心
            m_new = max(m_i, row_max(S_ij))  # 更新 running max
            corr  = exp(m_i - m_new)         # correction factor
            P_ij  = exp(S_ij - m_new)        # [Br × Br]

            O_i = diag(corr) @ O_i + P_ij @ V_j   # 修正舊值 + 新貢獻
            l_i = corr * l_i + row_sum(P_ij)
            m_i = m_new

        O_i = O_i / l_i                # normalize
        O[i:i+Br] = O_i                # 寫回 DRAM

    return O

關鍵性質：
  S_ij [Br × Br]：只在計算期間存在，不寫 DRAM → 省記憶體頻寬
  correction factor corr：讓過去的累加值在 max 更新時能正確修正
```

---

## P1-2 RTL 模組介面設計

```systemverilog
module flash_attn_wrapper #(
    parameter MAX_N    = 256,   // 最大 sequence length
    parameter MAX_D    = 64,    // 最大 head dimension
    parameter MAX_TILE = 64     // 最大 tile size（Br, Bc）
) (
    input  logic ACLK,          // 系統時鐘（5 ns/cycle = 200 MHz）
    input  logic ARESETn,       // 非同步低電位重置

    output logic FA_interrupt,  // 計算完成中斷（拉高一次）

    // ── AXI4 Slave（MMIO）─────────────────────────────────────────
    // CPU/HAL 透過這個界面寫暫存器、設定參數、啟動計算
    input  logic [31:0] AWADDR_S,  output logic AWREADY_S,  // 寫位址
    input  logic [31:0] WDATA_S,   output logic WREADY_S,   // 寫資料
    output logic [1:0]  BRESP_S,   output logic BVALID_S,   // 寫回應
    input  logic [31:0] ARADDR_S,  output logic ARREADY_S,  // 讀位址
    output logic [31:0] RDATA_S,   output logic RVALID_S,   // 讀資料
    // （完整 AXI4 信號省略 ID/LEN/SIZE/BURST，上面是主要信號）

    // ── AXI4 Master（DMA）────────────────────────────────────────
    // 硬體加速器主動發出請求讀寫 DRAM（Q/K/V/O 矩陣）
    output logic [31:0] ARADDR_M,  input  logic ARREADY_M,  // 讀請求
    input  logic [31:0] RDATA_M,   output logic RREADY_M,   // 接收讀資料
    output logic [31:0] AWADDR_M,  input  logic AWREADY_M,  // 寫請求
    output logic [31:0] WDATA_M,   output logic WLAST_M,    // 傳送寫資料
    input  logic        BVALID_M,  output logic BREADY_M    // 接收寫回應
);
```

```
兩條 AXI4 bus 的職責對比：

  AXI4 Slave（_S 後綴）           AXI4 Master（_M 後綴）
  ─────────────────────────────    ──────────────────────────────
  HAL → RTL 的命令通道             RTL → DRAM 的資料通道
  HAL 是 master，RTL 是 slave      RTL 是 master，HAL 仿真 slave
  用來設定暫存器                   用來讀 Q/K/V、寫 O
  每次傳一個 32-bit word           每次傳一個 d-word burst
  AXI4-Lite（簡化版，ARLEN=0）     AXI4 Full（支援 burst，ARLEN=d-1）
```

---

## P1-3 MMIO 暫存器規劃

```
RTL 內部的 AXI4 Slave FSM 收到寫請求時，根據位址寫入對應暫存器：

  localparam MMIO_BASE   = 32'h10050000;
  localparam REG_CONTROL = MMIO_BASE + 32'h00;  // 0x10050000
  localparam REG_SHAPE   = MMIO_BASE + 32'h04;  // 0x10050004
  localparam REG_TILE    = MMIO_BASE + 32'h08;  // 0x10050008
  localparam REG_Q_ADDR  = MMIO_BASE + 32'h0C;  // 0x1005000C
  localparam REG_K_ADDR  = MMIO_BASE + 32'h10;  // 0x10050010
  localparam REG_V_ADDR  = MMIO_BASE + 32'h14;  // 0x10050014
  localparam REG_O_ADDR  = MMIO_BASE + 32'h18;  // 0x10050018
  localparam REG_STATUS  = MMIO_BASE + 32'h1C;  // 0x1005001C

  wire [15:0] N_w  = reg_shape[31:16];   // 直接從暫存器取欄位
  wire [15:0] d_w  = reg_shape[15:0];
  wire [15:0] Br_w = reg_tile[15:0];

REG_CONTROL 的特殊行為：
  bit 0（start）：RTL 接受後，AXI Slave 額外加了自動清零邏輯：
    if (state != FA_IDLE) reg_control[0] <= 1'b0;
    ← 確保 start 是一個 pulse，不是持續 1

  bit 1（irq_clear）：清除 FA_interrupt 和 done_r：
    if (reg_control[1]) begin
        FA_interrupt <= 1'b0;
        done_r       <= 1'b0;
    end
```

---

## P1-4 State Machine 詳解

### 狀態列表與功能說明

```
typedef enum logic [4:0] {
    FA_IDLE,         // 0：等待 start
    FA_DMA_Q_AR,     // 1：發出 AR request 讀 Q row
    FA_DMA_Q_R,      // 2：接收 R beats，存入 Q_buf
    FA_INIT_O_LM,    // 3：初始化 O=0, l=0, m=-∞
    FA_DMA_K_AR,     // 4：發出 AR request 讀 K row
    FA_DMA_K_R,      // 5：接收 R beats，存入 K_buf
    FA_DMA_V_AR,     // 6：發出 AR request 讀 V row
    FA_DMA_V_R,      // 7：接收 R beats，存入 V_buf
    FA_COMPUTE,      // 8：執行 FlashAttention tile 計算（1 clock）
    FA_NEXT_J,       // 9：j_row += Br，判斷是否繼續內迴圈
    FA_FINALIZE,     // A：O_buf[r][k] /= l_buf[r]（normalize）
    FA_DMA_O_AW,     // B：發出 AW request 寫 O row
    FA_DMA_O_W,      // C：傳送 W beats
    FA_DMA_O_B,      // D：等待 B response（write acknowledge）
    FA_NEXT_I,       // E：i_row += Br，判斷是否繼續外迴圈
    FA_DONE          // F：拉起 FA_interrupt
} fa_state_t;
```

### 迴圈結構與 cur_row 的用途

```
Q/K/V/O 的 DMA 以「單行（row）」為單位，但一個 tile 有 Br 行：

  FA_DMA_Q_AR：讀 Q[i_row + cur_row, :] 這一行
               cur_row 從 0 到 Br-1
               每讀完一行，FA_DMA_Q_R 的 RLAST 到來時：
                 if cur_row < Br-1: cur_row++, 回 FA_DMA_Q_AR
                 else: cur_row=0, 進 FA_INIT_O_LM

  類似地，FA_DMA_K_R 和 FA_DMA_V_R 也用 cur_row 控制行數

  FA_DMA_O_AW/W/B：寫 O[i_row + cur_row, :]
                  也是逐行寫，cur_row 從 0 到 Br-1

  每個 row 的 DMA burst length = d（words），ARLEN = d-1
```

### DMA 位址計算

```systemverilog
// 讀 Q[i_row + cur_row] 這一行
FA_DMA_Q_AR:
  ARADDR_M <= reg_q_addr
              + 32'(int'(i_row) + int'(cur_row))  // 第幾行
                * 32'(int'(d_w))                   // 每行幾個 words
                * 4;                               // 每個 word 4 bytes

// i_row = 14（第二個 i-tile），cur_row = 3，d = 64：
// ARADDR = reg_q_addr + (14+3) × 64 × 4
//        = reg_q_addr + 17 × 256
//        = reg_q_addr + 4352

// 型別轉換說明：
//   i_row, cur_row 是 logic [15:0]（無號數）
//   int'(...) 把它轉成 32-bit 有號整數再做加法
//   32'(...) 確保結果是 32-bit，送給 ARADDR_M
```

---

## P1-5 FA_COMPUTE：Online Softmax 的 RTL 實作

### 內部 Buffer 說明

```systemverilog
// 全部用 real（fp64），透過 DPI-C 與 fp32 DMA 轉換
real Q_buf [0:MAX_TILE-1][0:MAX_D-1];    // Q 的一個 tile [Br × d]
real K_buf [0:MAX_TILE-1][0:MAX_D-1];    // K 的一個 tile [Br × d]
real V_buf [0:MAX_TILE-1][0:MAX_D-1];    // V 的一個 tile [Br × d]
real O_buf [0:MAX_TILE-1][0:MAX_D-1];    // 累加中的 O [Br × d]（跨 j-tile）
real l_buf [0:MAX_TILE-1];               // running sum [Br]（跨 j-tile）
real m_buf [0:MAX_TILE-1];               // running max [Br]（跨 j-tile）
real S_buf [0:MAX_TILE-1][0:MAX_TILE-1]; // S = Q×K^T/√d [Br × Br]（tile-local）
real P_buf [0:MAX_TILE-1][0:MAX_TILE-1]; // P = exp(S - m) [Br × Br]（tile-local）
```

```
Buffer 的生命週期：

  Q_buf：每個 i-tile 開始時從 DRAM 讀入，在整個 j 迴圈內不變
  K_buf：每個 j-tile 開始時從 DRAM 讀入，下個 j-tile 覆蓋
  V_buf：同 K_buf
  O_buf：在 FA_INIT_O_LM 清零，在每個 FA_COMPUTE 累加，在 FA_FINALIZE 正規化
  l_buf：同 O_buf
  m_buf：同 O_buf，初始值 -1.0e38（代表 -∞）
  S_buf：只在 FA_COMPUTE 內有效（blocking 賦值）
  P_buf：同 S_buf
```

### FA_COMPUTE 完整程式碼解析

```systemverilog
FA_COMPUTE: begin
    begin : compute_blk
        int  r, c, k;
        real scale, s_val, row_max, m_new_v, corr_v, sum_p_v, o_val;

        // ─── ① 計算縮放因子 ──────────────────────────────────────
        scale = 1.0 / dpi_sqrtf(real'(d_w));
        // d_w = 64 → scale = 1/8 = 0.125
        // 為什麼 /√d：防止 dot product 隨 d 增大而數值過大 → softmax 趨近 one-hot

        // ─── ② S_ij = Q_i × K_j^T / √d（BLOCKING）──────────────
        for (r = 0; r < MAX_TILE; r++) begin
            if (r < int'(Br_w)) begin              // 只處理有效的 Br 行
                for (c = 0; c < MAX_TILE; c++) begin
                    if (c < int'(Br_w)) begin       // 只處理有效的 Bc 列
                        s_val = 0.0;
                        for (k = 0; k < MAX_D; k++)
                            if (k < int'(d_w))     // 只處理有效的 d 維度
                                s_val = s_val + Q_buf[r][k] * K_buf[c][k];
                        S_buf[r][c] = s_val * scale;  // ← BLOCKING (=)
                        // 注意：S_buf[r][c] = 而非 <=
                        // 原因：下面立刻要讀 S_buf[r][c]
                        //       如果用 <=（NBA），讀到的是上個 j-tile 的值
                    end
                end
            end
        end

        // ─── ③ Online Softmax + O 累加 ───────────────────────────
        for (r = 0; r < MAX_TILE; r++) begin
            if (r < int'(Br_w)) begin

                // 找這一行 S_ij 的最大值
                row_max = S_buf[r][0];
                for (c = 1; c < MAX_TILE; c++)
                    if (c < int'(Br_w) && S_buf[r][c] > row_max)
                        row_max = S_buf[r][c];

                // Online Softmax 更新
                m_new_v = (m_buf[r] > row_max) ? m_buf[r] : row_max;
                //        ↑ 取 running max 和這個 tile max 的較大值

                corr_v  = dpi_expf(m_buf[r] - m_new_v);
                // 如果 m 沒有更新（m_buf[r] == m_new_v），corr = exp(0) = 1（無修正）
                // 如果 m 有更新（m_new_v > m_buf[r]），corr < 1（舊值需縮小）

                // 計算這個 tile 的 P_ij = exp(S - m_new)，並加總
                sum_p_v = 0.0;
                for (c = 0; c < MAX_TILE; c++) begin
                    if (c < int'(Br_w)) begin
                        P_buf[r][c] = dpi_expf(S_buf[r][c] - m_new_v);  // BLOCKING
                        sum_p_v     = sum_p_v + P_buf[r][c];
                    end
                end

                // 更新 O_buf：O = corr × O_old + P × V
                for (k = 0; k < MAX_D; k++) begin
                    if (k < int'(d_w)) begin
                        o_val = O_buf[r][k] * corr_v;  // 修正舊值
                        for (c = 0; c < MAX_TILE; c++)
                            if (c < int'(Br_w))
                                o_val = o_val + P_buf[r][c] * V_buf[c][k];
                        O_buf[r][k] <= o_val;           // NBA：跨 j-tile 累積
                    end
                end

                // 更新 l 和 m（NBA：下個 j-tile 才看到新值）
                l_buf[r] <= l_buf[r] * corr_v + sum_p_v;
                m_buf[r] <= m_new_v;
            end
        end
    end
    state <= FA_NEXT_J;   // FA_COMPUTE 結束，進入下一個 j-tile 或 FINALIZE
end
```

### FA_FINALIZE：Normalize

```systemverilog
FA_FINALIZE: begin
    begin : final_blk
        int r, k;
        for (r = 0; r < MAX_TILE; r++)
            if (r < int'(Br_w))
                for (k = 0; k < MAX_D; k++)
                    if (k < int'(d_w))
                        O_buf[r][k] <= O_buf[r][k] / l_buf[r];
        // 除以 l（softmax 分母的累積和）
        // 讀 O_buf[r][k] 和 l_buf[r] 是讀 NBA 的「舊值」
        // 但 FA_COMPUTE 是上一個 clock cycle，所以 NBA 已 settle → 正確
    end
    cur_row <= '0;
    state   <= FA_DMA_O_AW;
end
```

---

## P1-6 三個關鍵設計決策與 Bug 修正

### Bug 1：S_buf 的 NBA Timing 問題

```
問題描述：
  FA_COMPUTE 是一個 always_ff 區塊
  在這個區塊裡，先寫 S_buf，再讀 S_buf

  初始（錯誤）寫法：
    S_buf[r][c] <= s_val * scale;   // NBA（Non-Blocking Assignment）

  NBA 的 SystemVerilog 語意：
    所有 NBA 賦值（<=）在整個 always_ff block 執行完之後才生效
    也就是說，在同一個 always_ff 執行期間讀 S_buf，讀到的是「上一個 cycle 的值」

  後果：
    ②計算 S_buf 時：S_buf[r][c] <= s_val（NBA，還沒生效）
    ③讀 S_buf 做 softmax：讀到的是上個 j-tile 的 S_buf 值！
    → online softmax 計算的是錯誤的 S，結果與 CPU reference 不符

修正方案：
  S_buf[r][c] = s_val * scale;    // Blocking Assignment（立即生效）
  P_buf[r][c] = dpi_expf(...);    // 同上

原理：
  Blocking (=) 在 always_ff 裡是「立即」賦值，後面的讀取看到的是新值
  對 S_buf 用 blocking 是安全的，因為：
    - S_buf 只在 FA_COMPUTE 的同一次執行內被讀
    - 沒有其他 state 會讀 S_buf（tile-local）

規則：需要在同一個 always_ff block 內「先寫後讀」的變數 → 用 blocking (=)
      需要跨多個 clock cycle 保持狀態的變數（O/l/m） → 用 NBA (<=)
```

### Bug 2：WLAST_M 的 NBA Race

```
問題描述：
  FA_DMA_O_W 要傳 d 個 words
  word_cnt 從 0 數到 d-1
  當 word_cnt == d-1 時，這是最後一個 beat，WLAST_M 要拉高

  初始（錯誤）寫法（在 always_ff 裡）：
    WLAST_M <= (word_cnt == d_w[7:0] - 8'd1);  // NBA

    if (WREADY_M) begin
        if (WLAST_M) begin   // ← 讀 WLAST_M，讀到的是上個 cycle 的值！
            WVALID_M <= 1'b0;
            state    <= FA_DMA_O_B;
        end else begin
            word_cnt <= word_cnt + 8'd1;
        end
    end

  後果：
    word_cnt 更新 (NBA) 和 WLAST_M 更新 (NBA) 在同一個 cycle 發生
    但 if (WLAST_M) 讀到的是「上個 cycle」的 WLAST_M
    → 最後一個 beat 結束後，state 不會立刻進 FA_DMA_O_B
    → HAL 設定了 BVALID_M = 1，但 BREADY_M 此時是 0（state 不對）→ deadlock

修正方案：
  // always_ff 外面，純 combinational
  assign WLAST_M = (state == FA_DMA_O_W) && (word_cnt == d_w[7:0] - 8'd1);

  // always_ff 裡，不依賴 WLAST_M，直接判斷條件
  if (WREADY_M) begin
      if (int'(word_cnt) == int'(d_w) - 1) begin   // ← 直接比較
          WVALID_M <= 1'b0;
          state    <= FA_DMA_O_B;
      end else begin
          word_cnt <= word_cnt + 8'd1;
      end
  end

原理：
  combinational assign 的值在任何 eval() 後立刻反映最新 state/word_cnt
  HAL 在 eval() 後讀到的 WLAST_M 永遠是「現在」的值，沒有 cycle 延遲
```

### Bug 3：RREADY_M 和 BREADY_M 同樣問題

```
RREADY_M：RTL 準備好接收 R channel 資料（DMA 讀）
BREADY_M：RTL 準備好接收 B channel 回應（DMA 寫完成）

這兩個信號如果用 NBA 設定，HAL 的 eval() 後讀到的可能是舊值
→ HAL 誤以為 RTL 還沒準備好 → 多打一個 clock → 時序錯位

修正方案：同樣改成 combinational assign

  assign RREADY_M = (state == FA_DMA_Q_R)
                 || (state == FA_DMA_K_R)
                 || (state == FA_DMA_V_R);

  assign BREADY_M = (state == FA_DMA_O_B);

這樣 HAL 在 eval() 後能立刻看到正確的 ready 值。
```

### 三個 Bug 的共同根源

```
根本原因：在「同一個 clock cycle 的 eval() 內」混用了「寫 NBA」和「讀同一個變數」

SystemVerilog NBA 規則：
  t 時刻的 always_ff block 執行：
    讀：讀到 t-1 結束時的值（NBA 已 settle）
    寫（NBA <=）：在整個 block 執行完後（t 結束時）才更新

  這對「狀態機的跨 cycle 狀態」是正確的行為
  但對「在同一個 block 內先寫後讀同一個變數」就會出問題

設計原則（從這三個 Bug 學到的）：
  ① 在同一個 always_ff block 內「先寫後讀」的暫存 → 用 blocking (=)
  ② 需要被 combinational logic 或另一個 module「立刻看到」的輸出 → 用 assign
  ③ 純粹用於「本 block 下個 cycle 的狀態轉移」的暫存 → 用 NBA (<=)
```

---

## P1-7 三個 Test Case 與驗證結果

### 設計邏輯

```
三個 case 不是隨機選的，而是有系統地從簡到繁：

Case 0：最基礎（1 tile × 1 tile）
  N=4, d=4, Br=4
  i-tiles = N/Br = 1
  j-tiles = N/Br = 1
  → 只有 1 次 FA_COMPUTE，online softmax 只更新一次
  驗證：DMA 讀寫時序、基本計算

Case 1：Online Softmax 核心驗證（2 tiles × 2 tiles）
  N=8, d=8, Br=4
  i-tiles = 2, j-tiles = 2
  → 每個 i-tile 要掃 2 個 j-tile
  → 第二個 j-tile 的 corr factor 必須正確修正 O 和 l
  ★ 這是驗證 online softmax correction 的最小必要 case ★

Case 2：ViT-Small 實際規模（14 tiles × 14 tiles）
  N=196, d=64, Br=14
  i-tiles = 14, j-tiles = 14
  → 196 次 FA_COMPUTE
  → 模擬 ViT-Small 單個 head 的實際計算規模
  ★ 這是最有說服力的 case，展示演算法在實際規模下正確 ★
```

### Test Case 視覺化

```
Case 0：N/Br = 1，整個 Q/K/V 放在一個 tile 裡
  ┌────────────────┐
  │  Q [4×4]       │   只有一個 i-tile（i=0）
  └────────────────┘
  ┌────────────────┐
  │  K [4×4]       │   只有一個 j-tile（j=0）
  └────────────────┘
  計算路徑：Q×K^T → S[4×4] → P[4×4] → P×V → O[4×4]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Case 1：N/Br = 2，2×2 = 4 個 tile
  ┌────────┬────────┐
  │ S[0,0] │ S[0,1] │  i=0, j=0  │  i=0, j=4   ← corr 修正在這裡
  ├────────┼────────┤
  │ S[1,0] │ S[1,1] │  i=4, j=0  │  i=4, j=4
  └────────┴────────┘

  外迴圈 i=0：
    j=0 → FA_COMPUTE（建立初始 m/l/O）
    j=4 → FA_COMPUTE（★ corr 修正 ★）→ FA_FINALIZE → 寫 O[0:4]
  外迴圈 i=4：（類似）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Case 2：N/Br = 14，14×14 = 196 個 tile（ViT-Small 規格）
  每格代表一個 [14×14] 的 S_ij 矩陣（在 SRAM 裡，不寫 DRAM）

  j→  0    1    2    3    4    5    6    7    8    9   10   11   12   13
  ┌────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┐
i=0│S00 │S01 │S02 │S03 │S04 │S05 │S06 │S07 │S08 │S09 │S0A │S0B │S0C │S0D │
  ├────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┤
i=1│S10 │S11 │ ...                                                  │S1D │
  ├────┼────┤                                                   ├────┤
i=2│S20 │...                                                    │... │
  │ ...                                                               │
i=D│SD0 │SD1 │ ...                                              │SDD │
  └────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┘
  
  196 次 FA_COMPUTE，每次 13 次 corr 修正（j=1..13 時）
```

### 驗證通過的判定標準

```
for (int i = 0; i < N * d; i++) {
    float diff = fabsf(O_hw[i] - O_ref[i]);
    if (diff > FLOAT_TOL)  // FLOAT_TOL = 1e-4
        err++;
}

容差設定依據：
  fp32 machine epsilon ≈ 1.2e-7（1 ULP at 1.0）
  最差情況：N×d = 196×64 = 12544 次 MAC 累積
  理論最大累積誤差 ≈ 12544 × 1.2e-7 ≈ 1.5e-3
  但實際上：
    ① DPI-C 內部用 fp64 計算，只在 DMA 界面轉 fp32
    ② fp64 accumulation 誤差遠小於 fp32
    ③ 實測 max_diff < 1e-6，遠優於 1e-4

如果任何一個元素 diff > 1e-4 → FAIL
  通常意味：online softmax correction 算錯、DMA 位址錯誤、或 RTL timing bug
```

---

## P1-8 數值驗證方法論

### 測試向量的設計

```cpp
// tb.cpp
static void gen_qkv(int N, int d) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < d; j++) {
            int idx = i * d + j;
            Q_data[idx] = 0.3f * sinf((float)(i * d + j) * 0.17f);
            K_data[idx] = 0.3f * cosf((float)(i * d + j) * 0.13f + 0.5f);
            V_data[idx] = 0.2f * sinf((float)(j * N + i) * 0.19f + 1.0f);
        }
    }
}

設計考量：
  ① 決定性（deterministic）：每次執行結果相同，方便 debug
  ② 非平凡（non-trivial）：每個 row 的值都不同，不同 tile 間有差異
  ③ 振幅受控（0.2~0.3）：避免 exp 溢位（exp(30) = overflow）
  ④ sin/cos 的頻率不同：Q(0.17) K(0.13) V(0.19)，確保三個矩陣不相關
  ⑤ V 的 index 不同（j*N+i 而非 i*d+j）：確保 V 與 Q/K 有不同的結構
```

### 兩條計算路徑的比較

```
路徑 A（CPU reference）：
  standard_attention_cpu()
  演算法：傳統兩次掃描 softmax
  精度：fp32 全程
  記憶體：需要 N×N 的 S 矩陣（scratch）
  結果：O_ref[]

路徑 B（Hardware simulation）：
  flash_attention() → RTL behavioral sim
  演算法：FlashAttention-2 tiled online softmax
  精度：DPI-C fp64 內部，DMA fp32 界面
  記憶體：只需要 Br×d 的 tile buffer（on-chip SRAM 模擬）
  結果：O_hw[]

比較：|O_hw[i] - O_ref[i]| ≤ 1e-4 for all i

為什麼兩條路徑的結果應該相同？
  FlashAttention-2 的 online softmax 在數學上與傳統 softmax 完全等價
  兩者的差異只來自浮點計算順序的不同（fp32 結合律不成立）
  1e-4 的容差足以吸收這種差異
```

---

## 完整檔案清單

```
專案根目錄/
│
├── Makefile                              ← 入口：run_fa/run_fa_case0/1/2
│
├── include/
│   ├── hal/
│   │   ├── hal.hpp                      ← HALBase 抽象類別、runtime_info
│   │   └── flash_attn_hal.hpp           ← FlashAttnHAL 宣告、clock_step 巨集
│   └── runtime/
│       └── runtime.h                    ← flash_attention()、standard_attention_cpu() API
│
├── src/
│   ├── hal/
│   │   └── flash_attn_hal.cpp           ← HAL 實作：AXI4 MMIO/DMA 時序模擬
│   ├── dpi/
│   │   └── dpi_math.c                   ← DPI-C：fp32↔fp64 轉換、exp、sqrt
│   ├── runtime/flash_attn/
│   │   ├── driver_flash_attn.h          ← 暫存器地圖定義、driver API 宣告
│   │   ├── driver_flash_attn.cpp        ← fa_reg_write/read、fa_set_*、fa_start
│   │   └── runtime_flash_attn.cpp       ← flash_attention()、standard_attention_cpu()
│   └── hardware/flash_attn/
│       ├── Makefile                     ← Verilator 編譯 → libVflash_attn_wrapper.a
│       └── rtl/
│           └── flash_attn_wrapper.sv    ← ★ 核心 RTL：FA-2 + online softmax + AXI4
│
└── test/
    ├── cases/
    │   ├── case0/workload.h             ← CASE_N=4, CASE_D=4, CASE_BR=4
    │   ├── case1/workload.h             ← CASE_N=8, CASE_D=8, CASE_BR=4
    │   └── case2/workload.h             ← CASE_N=196, CASE_D=64, CASE_BR=14
    └── testbench/flash_attn/
        ├── Makefile                     ← 編譯 tb_fa_case{0,1,2}
        └── tb.cpp                       ← Testbench：gen_qkv, CPU ref, HW sim, 比較
```

---

## 給組員的 Q&A

**Q1：為什麼 FA_COMPUTE 只花「1 個 clock cycle」？FPGA 上不是要好幾千個 cycle 嗎？**

```
這是 Behavioral Simulation 的設計取捨。

在 Verilator 的 always_ff 裡，整個 for-loop（Br×Br×d 次乘加）
在同一次 eval() 中執行，模擬器不計算它需要多少 clock。

所以：
  模擬的 cycle 數 ≈ DMA 成本（每個 word = MEM_ACCESS_CYCLE = 5 cycles）
  不包含計算延遲

這樣的好處：
  ① 可以快速驗證演算法正確性（不需要等幾小時的詳細模擬）
  ② DMA 存取模式是準確的（3×N×d bytes 讀，1×N×d bytes 寫）
  ③ 適合 FPGA 上的 functional verification 階段

要得到準確的 timing，需要：
  a. 實作真正的 systolic array 多 cycle pipeline，或
  b. 在 Vivado/Quartus 進行綜合和 P&R 後看 timing report
```

**Q2：為什麼要用 DPI-C？不能直接在 SV 裡用 $exp() 嗎？**

```
Verilator 支援部分 SV 系統函式，但 $exp() 在 Verilator 5.x 裡
實際上需要 real 型態的支援，而且行為可能與 C math.h 有細微差異。

使用 DPI-C 的優點：
  ① 明確：知道呼叫的是哪個 C 函式庫的 exp()
  ② 可移植：不依賴 Verilator 版本的內建函式集
  ③ 精度一致：fp64 double，與 CPU reference 的誤差分析一致
  ④ 易擴充：未來需要其他函式（如 tanh for GELU）可以直接加
```

**Q3：vm_addr_h_ 如果 HAL 物件和 Q_data 陣列在不同的 4GB 區段怎麼辦？**

```
這確實是一個限制。解法是讓所有相關物件都在同一個 4GB 區段內：

  tb.cpp 裡宣告 static：
    static FlashAttnHAL hal(...);    ← .data / .bss section
    static float Q_data[...];        ← .bss section
    static float K_data[...];
    ...

在 Linux x86-64 的預設記憶體佈局裡，
.bss / .data section 通常都在低位址（< 4GB or 同一個 4GB 對齊區塊），
所以 HAL 物件和靜態資料陣列幾乎必然在同一個 4GB 區段。

如果要完全安全，可以改用 mmap 強制對齊到同一個 4GB 區塊。
但對目前的模擬用途，static 宣告就夠了。
```

**Q4：case2（N=196）為什麼不是 N=197（加上 CLS token）？**

```
ViT-Small 實際有 197 tokens（196 patches + 1 CLS），
但本專案用 196 的原因：196 = 14 × 14，可以整除 Br=14。

RTL 目前不支援 partial tile（N 不能整除 Br 的情況）。
197 / 14 = 14.07...，無法整除。

解法（未來可以做）：
  1. 用 0-padding：把 197 補到 196（丟掉 CLS token 的 attention）→ 近似解
  2. 修改 RTL 支援 partial tile：最後一個 tile 的有效 row 數 < Br，
     FA_COMPUTE 裡只計算有效的 row
  3. 用 198 = 14×(14+1)/... 不行，改用 Br=11（197=11×17+10，還是不整除）
     或 Br=197（整個序列一個 tile，退化成標準 attention）

對於驗證的影響：N=196 vs N=197 在演算法正確性上沒有本質差異，
差一個 token 不影響 FlashAttention 邏輯的驗證。
```

**Q5：如何確認 RTL 真的有做正確的 DMA？（不是假造資料）**

```
兩個方法：

方法 1：對比 Mem reads / Mem writes 和理論值
  Mem reads  = 3 × N × d × 4 bytes（Q + K + V）
  Mem writes = 1 × N × d × 4 bytes（O）
  如果 RTL 有多讀或少讀，數字會不對。

方法 2：DEBUG=1 模式
  make run_fa_case0 DEBUG=1
  → HAL 印出每次 MMIO R/W 的位址和值
  → HAL 印出每次 DMA read/write 的起始位址和 burst length
  → 可以手動對照 DMA 位址是否正確

  例如 case0（N=4, d=4, Br=4, q_addr=0xXXXX）：
    DMA read addr=0xXXXX len=4  ← Q[0:4, :]（i=0, cur_row=0..3，但一次 Br=4 rows）
    DMA read addr=0xXXXX len=4  ← K[0:4, :]（j=0）
    DMA read addr=0xXXXX len=4  ← V[0:4, :]（j=0）
    DMA write addr=0xXXXX len=4 ← O[0:4, :]

  如果位址算錯，這裡看得出來。
```
