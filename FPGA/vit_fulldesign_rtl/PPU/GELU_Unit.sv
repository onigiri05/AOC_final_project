`include "ASIC.svh"

module GELU_Unit (
    input  logic                         clk,
    input  logic                         rst_n,
    input  logic                         en,              // FC1 輸出時的致能信號
    input  logic signed [`DATA_BITS-1:0] data_in,         // 來自 FC1 的原始高精度輸出
    output logic signed [`DATA_BITS-1:0] data_out         // 輸出給 Requant Unit 的中間估計值
);

    logic [7:0] gelu_in_mapped;
    logic [7:0] gelu_lut_out;
    
    // 實體常數查找表 ROM (256 x 8-bit)
    logic [7:0] gelu_rom [0:255];
    
    // 映射排程：依據固定點軟體模擬範圍截取適當的 8 位元作為 LUT 索引
    assign gelu_in_mapped = data_in[15:8]; 

    // 初始化硬體 ROM 常數 (由 Python 腳本量化生成)
    initial begin
        gelu_rom[0] = 8'h00; // 索引   0 (signed    0) | x= 0.000, y= 0.000
        gelu_rom[1] = 8'h0F; // 索引   1 (signed    1) | x= 0.256, y= 0.154
        gelu_rom[2] = 8'h24; // 索引   2 (signed    2) | x= 0.512, y= 0.356
        gelu_rom[3] = 8'h3C; // 索引   3 (signed    3) | x= 0.768, y= 0.598
        gelu_rom[4] = 8'h57; // 索引   4 (signed    4) | x= 1.024, y= 0.867
        gelu_rom[5] = 8'h73; // 索引   5 (signed    5) | x= 1.280, y= 1.151
        gelu_rom[6] = 8'h90; // 索引   6 (signed    6) | x= 1.536, y= 1.440
        gelu_rom[7] = 8'hAD; // 索引   7 (signed    7) | x= 1.792, y= 1.726
        gelu_rom[8] = 8'hC9; // 索引   8 (signed    8) | x= 2.048, y= 2.007
        gelu_rom[9] = 8'hE4; // 索引   9 (signed    9) | x= 2.304, y= 2.280
        gelu_rom[10] = 8'hFF; // 索引  10 (signed   10) | x= 2.560, y= 2.547
        gelu_rom[11] = 8'hFF; // 索引  11 (signed   11) | x= 2.816, y= 2.810
        gelu_rom[12] = 8'hFF; // 索引  12 (signed   12) | x= 3.072, y= 3.069
        gelu_rom[13] = 8'hFF; // 索引  13 (signed   13) | x= 3.328, y= 3.327
        gelu_rom[14] = 8'hFF; // 索引  14 (signed   14) | x= 3.584, y= 3.584
        gelu_rom[15] = 8'hFF; // 索引  15 (signed   15) | x= 3.840, y= 3.840
        gelu_rom[16] = 8'hFF; // 索引  16 (signed   16) | x= 4.096, y= 4.096
        gelu_rom[17] = 8'hFF; // 索引  17 (signed   17) | x= 4.352, y= 4.352
        gelu_rom[18] = 8'hFF; // 索引  18 (signed   18) | x= 4.608, y= 4.608
        gelu_rom[19] = 8'hFF; // 索引  19 (signed   19) | x= 4.864, y= 4.864
        gelu_rom[20] = 8'hFF; // 索引  20 (signed   20) | x= 5.120, y= 5.120
        gelu_rom[21] = 8'hFF; // 索引  21 (signed   21) | x= 5.376, y= 5.376
        gelu_rom[22] = 8'hFF; // 索引  22 (signed   22) | x= 5.632, y= 5.632
        gelu_rom[23] = 8'hFF; // 索引  23 (signed   23) | x= 5.888, y= 5.888
        gelu_rom[24] = 8'hFF; // 索引  24 (signed   24) | x= 6.144, y= 6.144
        gelu_rom[25] = 8'hFF; // 索引  25 (signed   25) | x= 6.400, y= 6.400
        gelu_rom[26] = 8'hFF; // 索引  26 (signed   26) | x= 6.656, y= 6.656
        gelu_rom[27] = 8'hFF; // 索引  27 (signed   27) | x= 6.912, y= 6.912
        gelu_rom[28] = 8'hFF; // 索引  28 (signed   28) | x= 7.168, y= 7.168
        gelu_rom[29] = 8'hFF; // 索引  29 (signed   29) | x= 7.424, y= 7.424
        gelu_rom[30] = 8'hFF; // 索引  30 (signed   30) | x= 7.680, y= 7.680
        gelu_rom[31] = 8'hFF; // 索引  31 (signed   31) | x= 7.936, y= 7.936
        gelu_rom[32] = 8'hFF; // 索引  32 (signed   32) | x= 8.192, y= 8.192
        gelu_rom[33] = 8'hFF; // 索引  33 (signed   33) | x= 8.448, y= 8.448
        gelu_rom[34] = 8'hFF; // 索引  34 (signed   34) | x= 8.704, y= 8.704
        gelu_rom[35] = 8'hFF; // 索引  35 (signed   35) | x= 8.960, y= 8.960
        gelu_rom[36] = 8'hFF; // 索引  36 (signed   36) | x= 9.216, y= 9.216
        gelu_rom[37] = 8'hFF; // 索引  37 (signed   37) | x= 9.472, y= 9.472
        gelu_rom[38] = 8'hFF; // 索引  38 (signed   38) | x= 9.728, y= 9.728
        gelu_rom[39] = 8'hFF; // 索引  39 (signed   39) | x= 9.984, y= 9.984
        gelu_rom[40] = 8'hFF; // 索引  40 (signed   40) | x=10.240, y=10.240
        gelu_rom[41] = 8'hFF; // 索引  41 (signed   41) | x=10.496, y=10.496
        gelu_rom[42] = 8'hFF; // 索引  42 (signed   42) | x=10.752, y=10.752
        gelu_rom[43] = 8'hFF; // 索引  43 (signed   43) | x=11.008, y=11.008
        gelu_rom[44] = 8'hFF; // 索引  44 (signed   44) | x=11.264, y=11.264
        gelu_rom[45] = 8'hFF; // 索引  45 (signed   45) | x=11.520, y=11.520
        gelu_rom[46] = 8'hFF; // 索引  46 (signed   46) | x=11.776, y=11.776
        gelu_rom[47] = 8'hFF; // 索引  47 (signed   47) | x=12.032, y=12.032
        gelu_rom[48] = 8'hFF; // 索引  48 (signed   48) | x=12.288, y=12.288
        gelu_rom[49] = 8'hFF; // 索引  49 (signed   49) | x=12.544, y=12.544
        gelu_rom[50] = 8'hFF; // 索引  50 (signed   50) | x=12.800, y=12.800
        gelu_rom[51] = 8'hFF; // 索引  51 (signed   51) | x=13.056, y=13.056
        gelu_rom[52] = 8'hFF; // 索引  52 (signed   52) | x=13.312, y=13.312
        gelu_rom[53] = 8'hFF; // 索引  53 (signed   53) | x=13.568, y=13.568
        gelu_rom[54] = 8'hFF; // 索引  54 (signed   54) | x=13.824, y=13.824
        gelu_rom[55] = 8'hFF; // 索引  55 (signed   55) | x=14.080, y=14.080
        gelu_rom[56] = 8'hFF; // 索引  56 (signed   56) | x=14.336, y=14.336
        gelu_rom[57] = 8'hFF; // 索引  57 (signed   57) | x=14.592, y=14.592
        gelu_rom[58] = 8'hFF; // 索引  58 (signed   58) | x=14.848, y=14.848
        gelu_rom[59] = 8'hFF; // 索引  59 (signed   59) | x=15.104, y=15.104
        gelu_rom[60] = 8'hFF; // 索引  60 (signed   60) | x=15.360, y=15.360
        gelu_rom[61] = 8'hFF; // 索引  61 (signed   61) | x=15.616, y=15.616
        gelu_rom[62] = 8'hFF; // 索引  62 (signed   62) | x=15.872, y=15.872
        gelu_rom[63] = 8'hFF; // 索引  63 (signed   63) | x=16.128, y=16.128
        gelu_rom[64] = 8'hFF; // 索引  64 (signed   64) | x=16.384, y=16.384
        gelu_rom[65] = 8'hFF; // 索引  65 (signed   65) | x=16.640, y=16.640
        gelu_rom[66] = 8'hFF; // 索引  66 (signed   66) | x=16.896, y=16.896
        gelu_rom[67] = 8'hFF; // 索引  67 (signed   67) | x=17.152, y=17.152
        gelu_rom[68] = 8'hFF; // 索引  68 (signed   68) | x=17.408, y=17.408
        gelu_rom[69] = 8'hFF; // 索引  69 (signed   69) | x=17.664, y=17.664
        gelu_rom[70] = 8'hFF; // 索引  70 (signed   70) | x=17.920, y=17.920
        gelu_rom[71] = 8'hFF; // 索引  71 (signed   71) | x=18.176, y=18.176
        gelu_rom[72] = 8'hFF; // 索引  72 (signed   72) | x=18.432, y=18.432
        gelu_rom[73] = 8'hFF; // 索引  73 (signed   73) | x=18.688, y=18.688
        gelu_rom[74] = 8'hFF; // 索引  74 (signed   74) | x=18.944, y=18.944
        gelu_rom[75] = 8'hFF; // 索引  75 (signed   75) | x=19.200, y=19.200
        gelu_rom[76] = 8'hFF; // 索引  76 (signed   76) | x=19.456, y=19.456
        gelu_rom[77] = 8'hFF; // 索引  77 (signed   77) | x=19.712, y=19.712
        gelu_rom[78] = 8'hFF; // 索引  78 (signed   78) | x=19.968, y=19.968
        gelu_rom[79] = 8'hFF; // 索引  79 (signed   79) | x=20.224, y=20.224
        gelu_rom[80] = 8'hFF; // 索引  80 (signed   80) | x=20.480, y=20.480
        gelu_rom[81] = 8'hFF; // 索引  81 (signed   81) | x=20.736, y=20.736
        gelu_rom[82] = 8'hFF; // 索引  82 (signed   82) | x=20.992, y=20.992
        gelu_rom[83] = 8'hFF; // 索引  83 (signed   83) | x=21.248, y=21.248
        gelu_rom[84] = 8'hFF; // 索引  84 (signed   84) | x=21.504, y=21.504
        gelu_rom[85] = 8'hFF; // 索引  85 (signed   85) | x=21.760, y=21.760
        gelu_rom[86] = 8'hFF; // 索引  86 (signed   86) | x=22.016, y=22.016
        gelu_rom[87] = 8'hFF; // 索引  87 (signed   87) | x=22.272, y=22.272
        gelu_rom[88] = 8'hFF; // 索引  88 (signed   88) | x=22.528, y=22.528
        gelu_rom[89] = 8'hFF; // 索引  89 (signed   89) | x=22.784, y=22.784
        gelu_rom[90] = 8'hFF; // 索引  90 (signed   90) | x=23.040, y=23.040
        gelu_rom[91] = 8'hFF; // 索引  91 (signed   91) | x=23.296, y=23.296
        gelu_rom[92] = 8'hFF; // 索引  92 (signed   92) | x=23.552, y=23.552
        gelu_rom[93] = 8'hFF; // 索引  93 (signed   93) | x=23.808, y=23.808
        gelu_rom[94] = 8'hFF; // 索引  94 (signed   94) | x=24.064, y=24.064
        gelu_rom[95] = 8'hFF; // 索引  95 (signed   95) | x=24.320, y=24.320
        gelu_rom[96] = 8'hFF; // 索引  96 (signed   96) | x=24.576, y=24.576
        gelu_rom[97] = 8'hFF; // 索引  97 (signed   97) | x=24.832, y=24.832
        gelu_rom[98] = 8'hFF; // 索引  98 (signed   98) | x=25.088, y=25.088
        gelu_rom[99] = 8'hFF; // 索引  99 (signed   99) | x=25.344, y=25.344
        gelu_rom[100] = 8'hFF; // 索引 100 (signed  100) | x=25.600, y=25.600
        gelu_rom[101] = 8'hFF; // 索引 101 (signed  101) | x=25.856, y=25.856
        gelu_rom[102] = 8'hFF; // 索引 102 (signed  102) | x=26.112, y=26.112
        gelu_rom[103] = 8'hFF; // 索引 103 (signed  103) | x=26.368, y=26.368
        gelu_rom[104] = 8'hFF; // 索引 104 (signed  104) | x=26.624, y=26.624
        gelu_rom[105] = 8'hFF; // 索引 105 (signed  105) | x=26.880, y=26.880
        gelu_rom[106] = 8'hFF; // 索引 106 (signed  106) | x=27.136, y=27.136
        gelu_rom[107] = 8'hFF; // 索引 107 (signed  107) | x=27.392, y=27.392
        gelu_rom[108] = 8'hFF; // 索引 108 (signed  108) | x=27.648, y=27.648
        gelu_rom[109] = 8'hFF; // 索引 109 (signed  109) | x=27.904, y=27.904
        gelu_rom[110] = 8'hFF; // 索引 110 (signed  110) | x=28.160, y=28.160
        gelu_rom[111] = 8'hFF; // 索引 111 (signed  111) | x=28.416, y=28.416
        gelu_rom[112] = 8'hFF; // 索引 112 (signed  112) | x=28.672, y=28.672
        gelu_rom[113] = 8'hFF; // 索引 113 (signed  113) | x=28.928, y=28.928
        gelu_rom[114] = 8'hFF; // 索引 114 (signed  114) | x=29.184, y=29.184
        gelu_rom[115] = 8'hFF; // 索引 115 (signed  115) | x=29.440, y=29.440
        gelu_rom[116] = 8'hFF; // 索引 116 (signed  116) | x=29.696, y=29.696
        gelu_rom[117] = 8'hFF; // 索引 117 (signed  117) | x=29.952, y=29.952
        gelu_rom[118] = 8'hFF; // 索引 118 (signed  118) | x=30.208, y=30.208
        gelu_rom[119] = 8'hFF; // 索引 119 (signed  119) | x=30.464, y=30.464
        gelu_rom[120] = 8'hFF; // 索引 120 (signed  120) | x=30.720, y=30.720
        gelu_rom[121] = 8'hFF; // 索引 121 (signed  121) | x=30.976, y=30.976
        gelu_rom[122] = 8'hFF; // 索引 122 (signed  122) | x=31.232, y=31.232
        gelu_rom[123] = 8'hFF; // 索引 123 (signed  123) | x=31.488, y=31.488
        gelu_rom[124] = 8'hFF; // 索引 124 (signed  124) | x=31.744, y=31.744
        gelu_rom[125] = 8'hFF; // 索引 125 (signed  125) | x=32.000, y=32.000
        gelu_rom[126] = 8'hFF; // 索引 126 (signed  126) | x=32.256, y=32.256
        gelu_rom[127] = 8'hFF; // 索引 127 (signed  127) | x=32.512, y=32.512
        gelu_rom[128] = 8'h00; // 索引 128 (signed -128) | x=-32.768, y=-0.000
        gelu_rom[129] = 8'h00; // 索引 129 (signed -127) | x=-32.512, y=-0.000
        gelu_rom[130] = 8'h00; // 索引 130 (signed -126) | x=-32.256, y=-0.000
        gelu_rom[131] = 8'h00; // 索引 131 (signed -125) | x=-32.000, y=-0.000
        gelu_rom[132] = 8'h00; // 索引 132 (signed -124) | x=-31.744, y=-0.000
        gelu_rom[133] = 8'h00; // 索引 133 (signed -123) | x=-31.488, y=-0.000
        gelu_rom[134] = 8'h00; // 索引 134 (signed -122) | x=-31.232, y=-0.000
        gelu_rom[135] = 8'h00; // 索引 135 (signed -121) | x=-30.976, y=-0.000
        gelu_rom[136] = 8'h00; // 索引 136 (signed -120) | x=-30.720, y=-0.000
        gelu_rom[137] = 8'h00; // 索引 137 (signed -119) | x=-30.464, y=-0.000
        gelu_rom[138] = 8'h00; // 索引 138 (signed -118) | x=-30.208, y=-0.000
        gelu_rom[139] = 8'h00; // 索引 139 (signed -117) | x=-29.952, y=-0.000
        gelu_rom[140] = 8'h00; // 索引 140 (signed -116) | x=-29.696, y=-0.000
        gelu_rom[141] = 8'h00; // 索引 141 (signed -115) | x=-29.440, y=-0.000
        gelu_rom[142] = 8'h00; // 索引 142 (signed -114) | x=-29.184, y=-0.000
        gelu_rom[143] = 8'h00; // 索引 143 (signed -113) | x=-28.928, y=-0.000
        gelu_rom[144] = 8'h00; // 索引 144 (signed -112) | x=-28.672, y=-0.000
        gelu_rom[145] = 8'h00; // 索引 145 (signed -111) | x=-28.416, y=-0.000
        gelu_rom[146] = 8'h00; // 索引 146 (signed -110) | x=-28.160, y=-0.000
        gelu_rom[147] = 8'h00; // 索引 147 (signed -109) | x=-27.904, y=-0.000
        gelu_rom[148] = 8'h00; // 索引 148 (signed -108) | x=-27.648, y=-0.000
        gelu_rom[149] = 8'h00; // 索引 149 (signed -107) | x=-27.392, y=-0.000
        gelu_rom[150] = 8'h00; // 索引 150 (signed -106) | x=-27.136, y=-0.000
        gelu_rom[151] = 8'h00; // 索引 151 (signed -105) | x=-26.880, y=-0.000
        gelu_rom[152] = 8'h00; // 索引 152 (signed -104) | x=-26.624, y=-0.000
        gelu_rom[153] = 8'h00; // 索引 153 (signed -103) | x=-26.368, y=-0.000
        gelu_rom[154] = 8'h00; // 索引 154 (signed -102) | x=-26.112, y=-0.000
        gelu_rom[155] = 8'h00; // 索引 155 (signed -101) | x=-25.856, y=-0.000
        gelu_rom[156] = 8'h00; // 索引 156 (signed -100) | x=-25.600, y=-0.000
        gelu_rom[157] = 8'h00; // 索引 157 (signed  -99) | x=-25.344, y=-0.000
        gelu_rom[158] = 8'h00; // 索引 158 (signed  -98) | x=-25.088, y=-0.000
        gelu_rom[159] = 8'h00; // 索引 159 (signed  -97) | x=-24.832, y=-0.000
        gelu_rom[160] = 8'h00; // 索引 160 (signed  -96) | x=-24.576, y=-0.000
        gelu_rom[161] = 8'h00; // 索引 161 (signed  -95) | x=-24.320, y=-0.000
        gelu_rom[162] = 8'h00; // 索引 162 (signed  -94) | x=-24.064, y=-0.000
        gelu_rom[163] = 8'h00; // 索引 163 (signed  -93) | x=-23.808, y=-0.000
        gelu_rom[164] = 8'h00; // 索引 164 (signed  -92) | x=-23.552, y=-0.000
        gelu_rom[165] = 8'h00; // 索引 165 (signed  -91) | x=-23.296, y=-0.000
        gelu_rom[166] = 8'h00; // 索引 166 (signed  -90) | x=-23.040, y=-0.000
        gelu_rom[167] = 8'h00; // 索引 167 (signed  -89) | x=-22.784, y=-0.000
        gelu_rom[168] = 8'h00; // 索引 168 (signed  -88) | x=-22.528, y=-0.000
        gelu_rom[169] = 8'h00; // 索引 169 (signed  -87) | x=-22.272, y=-0.000
        gelu_rom[170] = 8'h00; // 索引 170 (signed  -86) | x=-22.016, y=-0.000
        gelu_rom[171] = 8'h00; // 索引 171 (signed  -85) | x=-21.760, y=-0.000
        gelu_rom[172] = 8'h00; // 索引 172 (signed  -84) | x=-21.504, y=-0.000
        gelu_rom[173] = 8'h00; // 索引 173 (signed  -83) | x=-21.248, y=-0.000
        gelu_rom[174] = 8'h00; // 索引 174 (signed  -82) | x=-20.992, y=-0.000
        gelu_rom[175] = 8'h00; // 索引 175 (signed  -81) | x=-20.736, y=-0.000
        gelu_rom[176] = 8'h00; // 索引 176 (signed  -80) | x=-20.480, y=-0.000
        gelu_rom[177] = 8'h00; // 索引 177 (signed  -79) | x=-20.224, y=-0.000
        gelu_rom[178] = 8'h00; // 索引 178 (signed  -78) | x=-19.968, y=-0.000
        gelu_rom[179] = 8'h00; // 索引 179 (signed  -77) | x=-19.712, y=-0.000
        gelu_rom[180] = 8'h00; // 索引 180 (signed  -76) | x=-19.456, y=-0.000
        gelu_rom[181] = 8'h00; // 索引 181 (signed  -75) | x=-19.200, y=-0.000
        gelu_rom[182] = 8'h00; // 索引 182 (signed  -74) | x=-18.944, y=-0.000
        gelu_rom[183] = 8'h00; // 索引 183 (signed  -73) | x=-18.688, y=-0.000
        gelu_rom[184] = 8'h00; // 索引 184 (signed  -72) | x=-18.432, y=-0.000
        gelu_rom[185] = 8'h00; // 索引 185 (signed  -71) | x=-18.176, y=-0.000
        gelu_rom[186] = 8'h00; // 索引 186 (signed  -70) | x=-17.920, y=-0.000
        gelu_rom[187] = 8'h00; // 索引 187 (signed  -69) | x=-17.664, y=-0.000
        gelu_rom[188] = 8'h00; // 索引 188 (signed  -68) | x=-17.408, y=-0.000
        gelu_rom[189] = 8'h00; // 索引 189 (signed  -67) | x=-17.152, y=-0.000
        gelu_rom[190] = 8'h00; // 索引 190 (signed  -66) | x=-16.896, y=-0.000
        gelu_rom[191] = 8'h00; // 索引 191 (signed  -65) | x=-16.640, y=-0.000
        gelu_rom[192] = 8'h00; // 索引 192 (signed  -64) | x=-16.384, y=-0.000
        gelu_rom[193] = 8'h00; // 索引 193 (signed  -63) | x=-16.128, y=-0.000
        gelu_rom[194] = 8'h00; // 索引 194 (signed  -62) | x=-15.872, y=-0.000
        gelu_rom[195] = 8'h00; // 索引 195 (signed  -61) | x=-15.616, y=-0.000
        gelu_rom[196] = 8'h00; // 索引 196 (signed  -60) | x=-15.360, y=-0.000
        gelu_rom[197] = 8'h00; // 索引 197 (signed  -59) | x=-15.104, y=-0.000
        gelu_rom[198] = 8'h00; // 索引 198 (signed  -58) | x=-14.848, y=-0.000
        gelu_rom[199] = 8'h00; // 索引 199 (signed  -57) | x=-14.592, y=-0.000
        gelu_rom[200] = 8'h00; // 索引 200 (signed  -56) | x=-14.336, y=-0.000
        gelu_rom[201] = 8'h00; // 索引 201 (signed  -55) | x=-14.080, y=-0.000
        gelu_rom[202] = 8'h00; // 索引 202 (signed  -54) | x=-13.824, y=-0.000
        gelu_rom[203] = 8'h00; // 索引 203 (signed  -53) | x=-13.568, y=-0.000
        gelu_rom[204] = 8'h00; // 索引 204 (signed  -52) | x=-13.312, y=-0.000
        gelu_rom[205] = 8'h00; // 索引 205 (signed  -51) | x=-13.056, y=-0.000
        gelu_rom[206] = 8'h00; // 索引 206 (signed  -50) | x=-12.800, y=-0.000
        gelu_rom[207] = 8'h00; // 索引 207 (signed  -49) | x=-12.544, y=-0.000
        gelu_rom[208] = 8'h00; // 索引 208 (signed  -48) | x=-12.288, y=-0.000
        gelu_rom[209] = 8'h00; // 索引 209 (signed  -47) | x=-12.032, y=-0.000
        gelu_rom[210] = 8'h00; // 索引 210 (signed  -46) | x=-11.776, y=-0.000
        gelu_rom[211] = 8'h00; // 索引 211 (signed  -45) | x=-11.520, y=-0.000
        gelu_rom[212] = 8'h00; // 索引 212 (signed  -44) | x=-11.264, y=-0.000
        gelu_rom[213] = 8'h00; // 索引 213 (signed  -43) | x=-11.008, y=-0.000
        gelu_rom[214] = 8'h00; // 索引 214 (signed  -42) | x=-10.752, y=-0.000
        gelu_rom[215] = 8'h00; // 索引 215 (signed  -41) | x=-10.496, y=-0.000
        gelu_rom[216] = 8'h00; // 索引 216 (signed  -40) | x=-10.240, y=-0.000
        gelu_rom[217] = 8'h00; // 索引 217 (signed  -39) | x=-9.984, y=-0.000
        gelu_rom[218] = 8'h00; // 索引 218 (signed  -38) | x=-9.728, y=-0.000
        gelu_rom[219] = 8'h00; // 索引 219 (signed  -37) | x=-9.472, y=-0.000
        gelu_rom[220] = 8'h00; // 索引 220 (signed  -36) | x=-9.216, y=-0.000
        gelu_rom[221] = 8'h00; // 索引 221 (signed  -35) | x=-8.960, y=-0.000
        gelu_rom[222] = 8'h00; // 索引 222 (signed  -34) | x=-8.704, y=-0.000
        gelu_rom[223] = 8'h00; // 索引 223 (signed  -33) | x=-8.448, y=-0.000
        gelu_rom[224] = 8'h00; // 索引 224 (signed  -32) | x=-8.192, y=-0.000
        gelu_rom[225] = 8'h00; // 索引 225 (signed  -31) | x=-7.936, y=-0.000
        gelu_rom[226] = 8'h00; // 索引 226 (signed  -30) | x=-7.680, y=-0.000
        gelu_rom[227] = 8'h00; // 索引 227 (signed  -29) | x=-7.424, y=-0.000
        gelu_rom[228] = 8'h00; // 索引 228 (signed  -28) | x=-7.168, y=-0.000
        gelu_rom[229] = 8'h00; // 索引 229 (signed  -27) | x=-6.912, y=-0.000
        gelu_rom[230] = 8'h00; // 索引 230 (signed  -26) | x=-6.656, y=-0.000
        gelu_rom[231] = 8'h00; // 索引 231 (signed  -25) | x=-6.400, y=-0.000
        gelu_rom[232] = 8'h00; // 索引 232 (signed  -24) | x=-6.144, y=-0.000
        gelu_rom[233] = 8'h00; // 索引 233 (signed  -23) | x=-5.888, y=-0.000
        gelu_rom[234] = 8'h00; // 索引 234 (signed  -22) | x=-5.632, y=-0.000
        gelu_rom[235] = 8'h00; // 索引 235 (signed  -21) | x=-5.376, y=-0.000
        gelu_rom[236] = 8'h00; // 索引 236 (signed  -20) | x=-5.120, y=-0.000
        gelu_rom[237] = 8'h00; // 索引 237 (signed  -19) | x=-4.864, y=-0.000
        gelu_rom[238] = 8'h00; // 索引 238 (signed  -18) | x=-4.608, y=-0.000
        gelu_rom[239] = 8'h00; // 索引 239 (signed  -17) | x=-4.352, y=-0.000
        gelu_rom[240] = 8'h00; // 索引 240 (signed  -16) | x=-4.096, y=-0.000
        gelu_rom[241] = 8'h00; // 索引 241 (signed  -15) | x=-3.840, y=-0.000
        gelu_rom[242] = 8'h00; // 索引 242 (signed  -14) | x=-3.584, y=-0.000
        gelu_rom[243] = 8'h00; // 索引 243 (signed  -13) | x=-3.328, y=-0.001
        gelu_rom[244] = 8'h00; // 索引 244 (signed  -12) | x=-3.072, y=-0.003
        gelu_rom[245] = 8'h00; // 索引 245 (signed  -11) | x=-2.816, y=-0.006
        gelu_rom[246] = 8'h00; // 索引 246 (signed  -10) | x=-2.560, y=-0.013
        gelu_rom[247] = 8'h00; // 索引 247 (signed   -9) | x=-2.304, y=-0.024
        gelu_rom[248] = 8'h00; // 索引 248 (signed   -8) | x=-2.048, y=-0.041
        gelu_rom[249] = 8'h00; // 索引 249 (signed   -7) | x=-1.792, y=-0.066
        gelu_rom[250] = 8'h00; // 索引 250 (signed   -6) | x=-1.536, y=-0.096
        gelu_rom[251] = 8'h00; // 索引 251 (signed   -5) | x=-1.280, y=-0.129
        gelu_rom[252] = 8'h00; // 索引 252 (signed   -4) | x=-1.024, y=-0.157
        gelu_rom[253] = 8'h00; // 索引 253 (signed   -3) | x=-0.768, y=-0.170
        gelu_rom[254] = 8'h00; // 索引 254 (signed   -2) | x=-0.512, y=-0.156
        gelu_rom[255] = 8'h00; // 索引 255 (signed   -1) | x=-0.256, y=-0.102
    end

    always_ff @(posedge clk) begin
        if (en) begin
            gelu_lut_out <= gelu_rom[gelu_in_mapped];
        end
    end
    
    // 將 8-bit 查表結果擴展回 INT32 格式，無縫對接後級的 Requant Unit
    // 注意：因為 GELU_OUT 必定是正數 (GELU 幾乎不為負)，此處直接以 0 擴充高位元
    assign data_out = {24'b0, gelu_lut_out};

endmodule
