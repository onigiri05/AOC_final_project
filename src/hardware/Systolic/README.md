# Ststolic array
包含以下四個module
```
Systolic
├── Act_fifo.v
├── Opsum_acc.v
├── PE_pack.v
└── Systolic.v
```
## Systolic.v
- Top level module, 所有對外接口都由Systolic.v包裝
- I/O spec
    1. Control signal & config
        ``` verilog
        input en,
        output reg module_ready,      // module able to use
        
        input [16:0] act_base_addr, // BRAM total: 4096 * 140 B, / 8B per word, 17' addr
        input [16:0] w_base_addr,
        input [6:0] k_tile_cnt //at most 96, at FC2
        ```
        - en
            - 需要執行GEMM時要給en = 1'b1
        - module_ready
            - Systolic array已經可以執行下個m-tile時, ready = 1'b1
            - en與ready同時為high時會開始執行一個m-tile GEMM
        - 其他配置
            - act_base_addr
                - 這個m-tile的第一組activation address
            - w_base_addr
                - 這個m-tile的第一組activation address
            - k_tile_cnt
                - 這個m-tile包含多少個k-tile
                - i.e.多少次16 by 16 GEMM累加
                - eg: patch embedding [196, 768] * [768, 384], 對應就是ceil(768/16) = 48
    2. To BRAM (直接存取BRAM)
        ```v
        output [16:0] act_bram_addr, 
        output [16:0] w_bram_addr,

        input act_bram_valid, 
        input w_bram_valid,

        input [127:0] act_bram_row, //input data
        input [127:0] w_bram_row, //input data
        ```
        - 考慮bram的 word配置成128'的狀況
    3. To PPU
        ```v
        output opsum_valid,
        output [31:0] opsum
        ```
        - valid同時會給一個opsum, 總共會有256個, 連續給256個cycle

## Other module
1. Act_fifo.v
2. Opsum_acc.v
    - 累加不同k - tile的psum
3. PE_pack.v
    - PE unit, 處理2個weight對一個activation的MAC
    - map到一個DSP
    - 總共用128個