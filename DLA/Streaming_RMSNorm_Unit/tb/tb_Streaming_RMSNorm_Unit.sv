`timescale 1ns/1ps
`include "../src/Streaming_RMSNorm_Unit.sv"
`include "../src/Streaming_RMSNorm_RowPacker.sv"
module tb_Streaming_RMSNorm_Unit;

    // ============================================================
    // Full-shape RMSNorm testbench
    //
    // This TB reads software-generated files:
    //   hardware_export/rmsnorm_vectors/synthetic_full_shape_197x384/x_input.mem
    //   hardware_export/rmsnorm_vectors/synthetic_full_shape_197x384/inv_rms.mem
    //   hardware_export/rmsnorm_vectors/synthetic_full_shape_197x384/gamma.mem
    //   hardware_export/rmsnorm_vectors/synthetic_full_shape_197x384/golden.mem
    //
    // Run directory must contain hardware_export/...
    // Or override path by plusarg:
    //   +MEM_DIR=/path/to/synthetic_full_shape_197x384
    // ============================================================

    localparam int TOKEN_NUM   = 197;
    localparam int CHANNEL_NUM = 384;
    localparam int TOTAL_ELEMS = TOKEN_NUM * CHANNEL_NUM;

    localparam int TOKEN_AW    = $clog2(TOKEN_NUM);
    localparam int CHANNEL_AW  = $clog2(CHANNEL_NUM);

    localparam int X_W         = 8;
    localparam int SCALE_W     = 16;
    localparam int FRAC        = 14;
    localparam int OUT_SHIFT   = 0;

    localparam int ENABLE_BACKPRESSURE = 0;
    localparam int VERBOSE_PASS = 0;
    localparam int MAX_CYCLES = TOTAL_ELEMS * 20 + 1000;

    logic clk;
    logic rst_n;

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

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

    logic        [SCALE_W-1:0] inv_rms_mem [0:TOKEN_NUM-1];
    logic signed [SCALE_W-1:0] gamma_mem   [0:CHANNEL_NUM-1];

    assign inv_rms_data = inv_rms_mem[inv_rms_addr];
    assign gamma_data   = gamma_mem[gamma_addr];

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

    logic signed [X_W-1:0] x_mem      [0:TOTAL_ELEMS-1];
    logic signed [X_W-1:0] golden_mem [0:TOTAL_ELEMS-1];

    string mem_dir;
    string x_path;
    string inv_rms_path;
    string gamma_path;
    string golden_path;

    task automatic check_file_exists(input string path);
        int fd;
        begin
            fd = $fopen(path, "r");
            if (fd == 0) begin
                $display("[FATAL] Cannot open file: %s", path);
                $display("[HINT ] Run simulation from the folder containing hardware_export/");
                $display("[HINT ] Or pass +MEM_DIR=/path/to/synthetic_full_shape_197x384");
                $fatal(1);
            end
            else begin
                $fclose(fd);
            end
        end
    endtask

    task automatic load_mem_files;
        begin
            if (!$value$plusargs("MEM_DIR=%s", mem_dir)) begin
                mem_dir = "../hardware_export/rmsnorm_vectors/synthetic_full_shape_197x384";
            end

            x_path       = {mem_dir, "/x_input.mem"};
            inv_rms_path = {mem_dir, "/inv_rms.mem"};
            gamma_path   = {mem_dir, "/gamma.mem"};
            golden_path  = {mem_dir, "/golden.mem"};

            $display("========================================");
            $display("Loading RMSNorm software-generated vectors");
            $display("MEM_DIR      = %s", mem_dir);
            $display("x_input.mem  = %s", x_path);
            $display("inv_rms.mem  = %s", inv_rms_path);
            $display("gamma.mem    = %s", gamma_path);
            $display("golden.mem   = %s", golden_path);
            $display("TOKEN_NUM    = %0d", TOKEN_NUM);
            $display("CHANNEL_NUM  = %0d", CHANNEL_NUM);
            $display("TOTAL_ELEMS  = %0d", TOTAL_ELEMS);
            $display("========================================");

            check_file_exists(x_path);
            check_file_exists(inv_rms_path);
            check_file_exists(gamma_path);
            check_file_exists(golden_path);

            $readmemh(x_path,       x_mem);
            $readmemh(inv_rms_path, inv_rms_mem);
            $readmemh(gamma_path,   gamma_mem);
            $readmemh(golden_path,  golden_mem);

            $display("[INFO] All .mem files loaded successfully.");
            $display("[INFO] x_mem[0]      = %0d", $signed(x_mem[0]));
            $display("[INFO] inv_rms[0]    = 0x%04h", inv_rms_mem[0]);
            $display("[INFO] gamma[0]      = %0d / 0x%04h", $signed(gamma_mem[0]), gamma_mem[0]);
            $display("[INFO] golden_mem[0] = %0d", $signed(golden_mem[0]));
        end
    endtask

    int send_idx;

    task automatic drive_input;
        begin
            send_idx = 0;

            wait (busy === 1'b1);

            x_valid = 1'b1;
            x_in    = x_mem[0];

            while (send_idx < TOTAL_ELEMS) begin
                @(posedge clk);

                if (x_valid && x_ready) begin
                    send_idx = send_idx + 1;

                    if (send_idx < TOTAL_ELEMS) begin
                        x_in = x_mem[send_idx];
                    end
                    else begin
                        x_valid = 1'b0;
                        x_in    = '0;
                    end
                end
            end

            $display("[INFO] Input driver finished. Sent %0d elements.", send_idx);
        end
    endtask

    int recv_idx;
    int recv_t;
    int recv_c;
    int error_count;

    task automatic check_output;
        begin
            recv_idx    = 0;
            error_count = 0;

            wait (busy === 1'b1);

            while (recv_idx < TOTAL_ELEMS) begin
                @(posedge clk);

                if (y_valid && y_ready) begin
                    recv_t = recv_idx / CHANNEL_NUM;
                    recv_c = recv_idx % CHANNEL_NUM;

                    if (^y_out === 1'bx) begin
                        $display("[ERROR] token=%0d channel=%0d y_out is X/Z, golden=%0d",
                                 recv_t, recv_c, $signed(golden_mem[recv_idx]));
                        error_count = error_count + 1;
                    end
                    else if (y_out !== golden_mem[recv_idx]) begin
                        $display("[ERROR] token=%0d channel=%0d idx=%0d y_out=%0d golden=%0d x=%0d inv_rms=0x%04h gamma=0x%04h",
                                 recv_t, recv_c, recv_idx,
                                 $signed(y_out), $signed(golden_mem[recv_idx]),
                                 $signed(x_mem[recv_idx]), inv_rms_mem[recv_t], gamma_mem[recv_c]);
                        error_count = error_count + 1;
                    end
                    else if (VERBOSE_PASS) begin
                        $display("[PASS] token=%0d channel=%0d y_out=%0d",
                                 recv_t, recv_c, $signed(y_out));
                    end

                    if ((recv_idx != TOTAL_ELEMS-1) && y_last) begin
                        $display("[ERROR] y_last asserted too early at token=%0d channel=%0d idx=%0d",
                                 recv_t, recv_c, recv_idx);
                        error_count = error_count + 1;
                    end

                    if ((recv_idx == TOTAL_ELEMS-1) && !y_last) begin
                        $display("[ERROR] last output should assert y_last at final idx=%0d", recv_idx);
                        error_count = error_count + 1;
                    end

                    if ((recv_idx % 4096) == 0) begin
                        $display("[INFO] Checked %0d / %0d outputs", recv_idx, TOTAL_ELEMS);
                    end

                    recv_idx = recv_idx + 1;
                end
            end

            $display("[INFO] Output checker finished. Checked %0d elements.", recv_idx);
        end
    endtask

    initial begin
        y_ready = 1'b1;

        forever begin
            @(posedge clk);

            if (ENABLE_BACKPRESSURE) begin
                if ($time > 100 && (($time / 10) % 17 == 0))
                    y_ready <= 1'b0;
                else
                    y_ready <= 1'b1;
            end
            else begin
                y_ready <= 1'b1;
            end
        end
    end

    initial begin
        repeat (MAX_CYCLES) @(posedge clk);
        $display("[FATAL] Simulation timeout after %0d cycles", MAX_CYCLES);
        $display("[DEBUG] busy=%0b done=%0b send_idx=%0d recv_idx=%0d x_valid=%0b x_ready=%0b y_valid=%0b y_ready=%0b y_last=%0b",
                 busy, done, send_idx, recv_idx, x_valid, x_ready, y_valid, y_ready, y_last);
        $fatal(1);
    end

    initial begin
        load_mem_files();

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

        repeat (5) @(posedge clk);

        if (done !== 1'b1) begin
            $display("[WARN] done is not high at final check. This may be fine if the pulse already passed.");
        end

        if (error_count == 0) begin
            $display("========================================");
            $display("Streaming RMSNorm Unit TEST PASSED");
            $display("Checked elements = %0d", TOTAL_ELEMS);
            $display("========================================");
        end
        else begin
            $display("========================================");
            $display("Streaming RMSNorm Unit TEST FAILED, errors = %0d", error_count);
            $display("Checked elements = %0d", TOTAL_ELEMS);
            $display("========================================");
        end

        $finish;
    end

endmodule
