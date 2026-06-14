`timescale 1ns/1ps

module tb_Streaming_RMSNorm_Unit;

    // ============================================================
    // 測試參數
    // 為了模擬快一點，先用小尺寸測功能。
    // 確認正確後，再改成 TOKEN_NUM=197, CHANNEL_NUM=384。
    // ============================================================
    localparam int TOKEN_NUM   = 4;
    localparam int CHANNEL_NUM = 8;

    localparam int TOKEN_AW    = $clog2(TOKEN_NUM);
    localparam int CHANNEL_AW  = $clog2(CHANNEL_NUM);

    localparam int X_W         = 8;
    localparam int SCALE_W     = 16;
    localparam int FRAC        = 14;
    localparam int OUT_SHIFT   = 0;
    localparam int SHIFT       = 2 * FRAC + OUT_SHIFT;

    // ============================================================
    // clock / reset
    // ============================================================
    logic clk;
    logic rst_n;

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;   // 100 MHz
    end

    // ============================================================
    // DUT signals
    // ============================================================
    logic start;
    logic busy;
    logic done;

    logic x_valid;
    logic x_ready;
    logic signed [X_W-1:0] x_in;

    logic [TOKEN_AW-1:0] inv_rms_addr;
    logic [SCALE_W-1:0]  inv_rms_data;

    logic [CHANNEL_AW-1:0] gamma_addr;
    logic signed [SCALE_W-1:0] gamma_data;

    logic y_valid;
    logic y_ready;
    logic y_last;
    logic signed [X_W-1:0] y_out;

    // ============================================================
    // 模擬 Token Stat SRAM / Gamma Buffer
    // ============================================================
    logic [SCALE_W-1:0] inv_rms_mem [0:TOKEN_NUM-1];
    logic signed [SCALE_W-1:0] gamma_mem [0:CHANNEL_NUM-1];

    // 這裡先用 combinational read。
    // 如果未來接同步 BRAM，RTL wrapper 要多加 1-cycle 對齊。
    assign inv_rms_data = inv_rms_mem[inv_rms_addr];
    assign gamma_data   = gamma_mem[gamma_addr];

    // ============================================================
    // DUT
    // ============================================================
    Streaming_RMSNorm_Unit #(
        .TOKEN_NUM(TOKEN_NUM),
        .CHANNEL_NUM(CHANNEL_NUM),
        .TOKEN_AW(TOKEN_AW),
        .CHANNEL_AW(CHANNEL_AW),
        .X_W(X_W),
        .SCALE_W(SCALE_W),
        .FRAC(FRAC),
        .OUT_SHIFT(OUT_SHIFT)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),

        .start(start),
        .busy(busy),
        .done(done),

        .x_valid(x_valid),
        .x_ready(x_ready),
        .x_in(x_in),

        .inv_rms_addr(inv_rms_addr),
        .inv_rms_data(inv_rms_data),

        .gamma_addr(gamma_addr),
        .gamma_data(gamma_data),

        .y_valid(y_valid),
        .y_ready(y_ready),
        .y_last(y_last),
        .y_out(y_out)
    );

    // ============================================================
    // input activation 與 golden answer
    // ============================================================
    logic signed [X_W-1:0] x_mem [0:TOKEN_NUM-1][0:CHANNEL_NUM-1];
    logic signed [X_W-1:0] golden_mem [0:TOKEN_NUM-1][0:CHANNEL_NUM-1];

    int send_t, send_c;
    int recv_t, recv_c;
    int error_count;

    // ============================================================
    // INT8 clamp function
    // ============================================================
    function automatic signed [7:0] clamp_int8(input signed [63:0] v);
        begin
            if (v > 127)
                clamp_int8 = 8'sd127;
            else if (v < -128)
                clamp_int8 = -8'sd128;
            else
                clamp_int8 = v[7:0];
        end
    endfunction

    // ============================================================
    // golden 計算
    // ============================================================
    task automatic build_test_data;
        integer t, c;
        logic signed [63:0] prod;
        begin
            // inv_rms 使用 unsigned Q?.14
            // 1.0 = 16384
            // 這裡故意讓不同 token 有不同 inv_rms
            inv_rms_mem[0] = 16'd16384;  // 1.00
            inv_rms_mem[1] = 16'd8192;   // 0.50
            inv_rms_mem[2] = 16'd24576;  // 1.50
            inv_rms_mem[3] = 16'd32768;  // 2.00

            // gamma 使用 signed Q?.14
            // 1.0 = 16384
            for (c = 0; c < CHANNEL_NUM; c = c + 1) begin
                if (c[0] == 1'b0)
                    gamma_mem[c] = 16'sd16384;   // +1.0
                else
                    gamma_mem[c] = 16'sd8192;    // +0.5
            end

            // 建立 x 和 golden
            for (t = 0; t < TOKEN_NUM; t = t + 1) begin
                for (c = 0; c < CHANNEL_NUM; c = c + 1) begin
                    x_mem[t][c] = $signed((t * 11 + c * 7) % 128);

                    // 加一些負數測試
                    if ((t + c) % 3 == 0)
                        x_mem[t][c] = -x_mem[t][c];

                    prod = $signed(x_mem[t][c])
                         * $signed({1'b0, inv_rms_mem[t]})
                         * $signed(gamma_mem[c]);

                    golden_mem[t][c] = clamp_int8(prod >>> SHIFT);
                end
            end
        end
    endtask

    // ============================================================
    // input driver
    // 依照 token-major 順序送：
    // token0 ch0~chN, token1 ch0~chN ...
    // ============================================================
    task automatic drive_input;
        begin
            send_t = 0;
            send_c = 0;

            x_valid = 1'b1;
            x_in    = x_mem[0][0];

            while (send_t < TOKEN_NUM) begin
                @(posedge clk);

                if (x_valid && x_ready) begin
                    if (send_c == CHANNEL_NUM-1) begin
                        send_c = 0;
                        send_t = send_t + 1;
                    end
                    else begin
                        send_c = send_c + 1;
                    end

                    if (send_t < TOKEN_NUM) begin
                        x_in = x_mem[send_t][send_c];
                    end
                    else begin
                        x_valid = 1'b0;
                        x_in    = '0;
                    end
                end
            end
        end
    endtask

    // ============================================================
    // output checker
    // ============================================================
    task automatic check_output;
        begin
            recv_t = 0;
            recv_c = 0;
            error_count = 0;

            while (recv_t < TOKEN_NUM) begin
                @(posedge clk);

                if (y_valid && y_ready) begin
                    if (y_out !== golden_mem[recv_t][recv_c]) begin
                        $display("[ERROR] token=%0d channel=%0d y_out=%0d golden=%0d",
                                 recv_t, recv_c, y_out, golden_mem[recv_t][recv_c]);
                        error_count = error_count + 1;
                    end
                    else begin
                        $display("[PASS] token=%0d channel=%0d y_out=%0d",
                                 recv_t, recv_c, y_out);
                    end

                    if ((recv_t == TOKEN_NUM-1) && (recv_c == CHANNEL_NUM-1)) begin
                        if (!y_last) begin
                            $display("[ERROR] last output should assert y_last");
                            error_count = error_count + 1;
                        end
                    end

                    if (recv_c == CHANNEL_NUM-1) begin
                        recv_c = 0;
                        recv_t = recv_t + 1;
                    end
                    else begin
                        recv_c = recv_c + 1;
                    end
                end
            end
        end
    endtask

    // ============================================================
    // backpressure 測試
    // ============================================================
    initial begin
        y_ready = 1'b1;
        forever begin
            @(posedge clk);

            // 偶爾讓下游 not ready，測 valid/ready 是否能 stall
            if ($time > 100 && ($time % 70 == 0))
                y_ready <= 1'b0;
            else
                y_ready <= 1'b1;
        end
    end

    // ============================================================
    // main test
    // ============================================================
    initial begin
        build_test_data();

        rst_n   = 1'b0;
        start   = 1'b0;
        x_valid = 1'b0;
        x_in    = '0;

        repeat (5) @(posedge clk);
        rst_n = 1'b1;

        repeat (3) @(posedge clk);
        start = 1'b1;
        @(posedge clk);
        start = 1'b0;

        fork
            drive_input();
            check_output();
        join

        wait(done);
        repeat (5) @(posedge clk);

        if (error_count == 0) begin
            $display("========================================");
            $display("Streaming RMSNorm Unit TEST PASSED");
            $display("========================================");
        end
        else begin
            $display("========================================");
            $display("Streaming RMSNorm Unit TEST FAILED, errors = %0d", error_count);
            $display("========================================");
        end

        $finish;
    end

endmodule
