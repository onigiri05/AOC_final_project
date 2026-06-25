`timescale 1ns/1ps

module tb_vit_fulldesign_compile;
    localparam int IMG_WORDS            = 37632;
    localparam int POS_WORDS            = 18912;
    localparam int CLS_WORDS            = 96;
    localparam int GAMMA_WORDS          = 768;
    localparam int PATCH_WEIGHT_WORDS   = 73728;
    localparam int PATCH_BIAS_WORDS     = 384;
    localparam int TRANS_WEIGHT_WORDS   = 180224;
    localparam int TRANS_BIAS_WORDS     = 2048;
    localparam int X_WORDS              = 18912;
    localparam int GELU_DDR_WORDS       = 76800;
    localparam int PAGE_WORDS           = 1024;
    localparam int WEIGHT_TILE_WORDS    = 16;
    localparam int BIAS_TILE_WORDS      = 8;

    localparam int T_IMAGE      = 0;
    localparam int T_POS        = 1;
    localparam int T_CLS        = 2;
    localparam int T_GAMMA      = 4;
    localparam int T_PW_TILE    = 5;
    localparam int T_PB_TILE    = 6;
    localparam int T_TW_TILE    = 7;
    localparam int T_TB_TILE    = 8;
    localparam int T_GELU_PAGE  = 9;
    localparam int T_X_BUF      = 10;

    localparam int REG_CTRL     = 4'h0;
    localparam int REG_BASE     = 4'h4;
    localparam int REG_COUNT    = 4'h8;
    localparam int REG_DATA     = 4'hc;

    localparam int STATUS_BUSY             = 0;
    localparam int STATUS_DONE_STICKY      = 2;
    localparam int STATUS_INPUT_BUSY       = 3;
    localparam int STATUS_INPUT_ERROR      = 5;
    localparam int STATUS_PATCH_W_MISS     = 6;
    localparam int STATUS_PATCH_B_MISS     = 7;
    localparam int STATUS_TRANS_W_MISS     = 8;
    localparam int STATUS_TRANS_B_MISS     = 9;

    localparam longint unsigned MAX_CYCLES      = 64'd2000000000;
    localparam longint unsigned PROGRESS_CYCLES = 64'd1000000;

    string vec_dir;
    bit verbose_tile;

    logic [31:0] image_mem        [0:IMG_WORDS-1];
    logic [31:0] pos_mem          [0:POS_WORDS-1];
    logic [31:0] cls_mem          [0:CLS_WORDS-1];
    logic [31:0] gamma_mem        [0:GAMMA_WORDS-1];
    logic [31:0] patch_weight_mem [0:PATCH_WEIGHT_WORDS-1];
    logic [31:0] patch_bias_mem   [0:PATCH_BIAS_WORDS-1];
    logic [31:0] trans_weight_mem [0:TRANS_WEIGHT_WORDS-1];
    logic [31:0] trans_bias_mem   [0:TRANS_BIAS_WORDS-1];
    logic [31:0] x_out_gold_mem   [0:X_WORDS-1];
    logic [31:0] gelu_ddr_mem     [0:GELU_DDR_WORDS-1];

    logic clk;
    logic rst_n;
    logic start;
    logic clear_done;
    logic transformer_only;
    logic [5:0] patch_requant_shift;
    logic busy;
    logic done_pulse;
    logic done_sticky;

    logic input_wr_valid;
    logic input_wr_ready;
    logic [3:0] input_wr_addr;
    logic [31:0] input_wr_data;
    logic [3:0] input_wr_strb;
    logic input_busy;
    logic input_done;
    logic input_error;
    logic [3:0] input_active_target;
    logic [20:0] input_word_count;

    logic output_host_en;
    logic [3:0] output_host_target;
    logic [16:0] output_host_addr;
    logic [31:0] output_host_rd_data;

    logic stage_checkpoint_enable;
    logic stage_checkpoint_resume;
    logic stage_checkpoint_pending;
    logic stage_done_pulse;
    logic [3:0] stage_id;
    logic [4:0] stage_phase;
    logic gelu_store_done_i;

    logic [31:0] status_word;
    logic [31:0] patch_request_word;
    logic [31:0] trans_request_word;
    logic [31:0] page_request_word;
    logic [31:0] debug_word;
    logic [7:0] perf_bram_rd_words_o;
    logic [7:0] perf_bram_wr_words_o;
    logic perf_bram_active_o;
    logic [15:0] perf_mac_ops_o;
    logic perf_pingpong_wait_o;
    logic perf_pingpong_load_o;
    logic perf_pingpong_overlap_o;

    longint unsigned cycle_count;
    longint unsigned mac_count;
    longint unsigned bram_rd_words;
    longint unsigned bram_wr_words;
    longint unsigned bram_active_cycles;
    longint unsigned pingpong_wait_cycles;
    longint unsigned pingpong_load_cycles;
    longint unsigned pingpong_overlap_cycles;

    ViT_System_Core dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .clear_done(clear_done),
        .transformer_only(transformer_only),
        .patch_requant_shift(patch_requant_shift),
        .busy(busy),
        .done_pulse(done_pulse),
        .done_sticky(done_sticky),
        .input_wr_valid(input_wr_valid),
        .input_wr_ready(input_wr_ready),
        .input_wr_addr(input_wr_addr),
        .input_wr_data(input_wr_data),
        .input_wr_strb(input_wr_strb),
        .input_busy(input_busy),
        .input_done(input_done),
        .input_error(input_error),
        .input_active_target(input_active_target),
        .input_word_count(input_word_count),
        .output_host_en(output_host_en),
        .output_host_target(output_host_target),
        .output_host_addr(output_host_addr),
        .output_host_rd_data(output_host_rd_data),
        .stage_checkpoint_enable(stage_checkpoint_enable),
        .stage_checkpoint_resume(stage_checkpoint_resume),
        .stage_checkpoint_pending(stage_checkpoint_pending),
        .stage_done_pulse(stage_done_pulse),
        .stage_id(stage_id),
        .stage_phase(stage_phase),
        .gelu_store_done_i(gelu_store_done_i),
        .status_word(status_word),
        .patch_request_word(patch_request_word),
        .trans_request_word(trans_request_word),
        .page_request_word(page_request_word),
        .debug_word(debug_word),
        .perf_bram_rd_words_o(perf_bram_rd_words_o),
        .perf_bram_wr_words_o(perf_bram_wr_words_o),
        .perf_bram_active_o(perf_bram_active_o),
        .perf_mac_ops_o(perf_mac_ops_o),
        .perf_pingpong_wait_o(perf_pingpong_wait_o),
        .perf_pingpong_load_o(perf_pingpong_load_o),
        .perf_pingpong_overlap_o(perf_pingpong_overlap_o)
    );

    always #5 clk = ~clk;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cycle_count <= 64'd0;
            mac_count <= 64'd0;
            bram_rd_words <= 64'd0;
            bram_wr_words <= 64'd0;
            bram_active_cycles <= 64'd0;
            pingpong_wait_cycles <= 64'd0;
            pingpong_load_cycles <= 64'd0;
            pingpong_overlap_cycles <= 64'd0;
        end
        else begin
            cycle_count <= cycle_count + 64'd1;
            mac_count <= mac_count + perf_mac_ops_o;
            bram_rd_words <= bram_rd_words + perf_bram_rd_words_o;
            bram_wr_words <= bram_wr_words + perf_bram_wr_words_o;
            bram_active_cycles <= bram_active_cycles + perf_bram_active_o;
            pingpong_wait_cycles <= pingpong_wait_cycles + perf_pingpong_wait_o;
            pingpong_load_cycles <= pingpong_load_cycles + perf_pingpong_load_o;
            pingpong_overlap_cycles <= pingpong_overlap_cycles + perf_pingpong_overlap_o;
        end
    end

    function automatic [31:0] source_word(input int target, input int idx);
        begin
            source_word = 32'd0;
            case (target)
                T_IMAGE: begin
                    if ((idx >= 0) && (idx < IMG_WORDS)) source_word = image_mem[idx];
                end
                T_POS: begin
                    if ((idx >= 0) && (idx < POS_WORDS)) source_word = pos_mem[idx];
                end
                T_CLS: begin
                    if ((idx >= 0) && (idx < CLS_WORDS)) source_word = cls_mem[idx];
                end
                T_GAMMA: begin
                    if ((idx >= 0) && (idx < GAMMA_WORDS)) source_word = gamma_mem[idx];
                end
                T_PW_TILE: begin
                    if ((idx >= 0) && (idx < PATCH_WEIGHT_WORDS)) source_word = patch_weight_mem[idx];
                end
                T_PB_TILE: begin
                    if ((idx >= 0) && (idx < PATCH_BIAS_WORDS)) source_word = patch_bias_mem[idx];
                end
                T_TW_TILE: begin
                    if ((idx >= 0) && (idx < TRANS_WEIGHT_WORDS)) source_word = trans_weight_mem[idx];
                end
                T_TB_TILE: begin
                    if ((idx >= 0) && (idx < TRANS_BIAS_WORDS)) source_word = trans_bias_mem[idx];
                end
                T_GELU_PAGE: begin
                    if ((idx >= 0) && (idx < GELU_DDR_WORDS)) source_word = gelu_ddr_mem[idx];
                end
                default: source_word = 32'd0;
            endcase
        end
    endfunction

    task automatic host_write(input [3:0] addr, input [31:0] data, input [3:0] strb);
        begin
            @(posedge clk);
            input_wr_addr <= addr;
            input_wr_data <= data;
            input_wr_strb <= strb;
            input_wr_valid <= 1'b1;
            while (!input_wr_ready) begin
                @(posedge clk);
            end
            input_wr_valid <= 1'b0;
            input_wr_addr <= 4'd0;
            input_wr_data <= 32'd0;
            input_wr_strb <= 4'd0;
        end
    endtask

    task automatic wait_input_done_or_fail(input string label);
        begin
            while (!input_done) begin
                if (input_error) begin
                    $fatal(1, "[FAIL] input loader error while %s status=0x%08x", label, status_word);
                end
                @(posedge clk);
            end
            @(posedge clk);
        end
    endtask

    task automatic load_target_from_source(
        input int target,
        input int base,
        input int count,
        input int source_base,
        input string label
    );
        int i;
        begin
            if (count <= 0) begin
                $fatal(1, "[FAIL] zero/negative load count for %s target=%0d base=%0d", label, target, base);
            end
            host_write(REG_BASE, base, 4'hf);
            host_write(REG_COUNT, count, 4'hf);
            host_write(REG_CTRL, ((target << 4) | 1), 4'hf);
            for (i = 0; i < count; i = i + 1) begin
                host_write(REG_DATA, source_word(target, source_base + i), 4'hf);
            end
            wait_input_done_or_fail(label);
        end
    endtask

    task automatic read_host_word(input int target, input int addr, output logic [31:0] data);
        begin
            @(posedge clk);
            output_host_target <= target[3:0];
            output_host_addr <= addr[16:0];
            output_host_en <= 1'b1;
            @(posedge clk);
            output_host_en <= 1'b0;
            @(posedge clk);
            @(posedge clk);
            data = output_host_rd_data;
        end
    endtask

    task automatic service_gelu_store(input int base);
        int i;
        int count;
        logic [31:0] data;
        begin
            count = PAGE_WORDS;
            if ((base + count) > GELU_DDR_WORDS) count = GELU_DDR_WORDS - base;
            if (count < 0) count = 0;
            $display("[TB] GELU store page base=%0d count=%0d at cycle=%0d", base, count, cycle_count);
            for (i = 0; i < count; i = i + 1) begin
                read_host_word(T_GELU_PAGE, i, data);
                gelu_ddr_mem[base + i] = data;
            end
            @(posedge clk);
            gelu_store_done_i <= 1'b1;
            @(posedge clk);
            gelu_store_done_i <= 1'b0;
        end
    endtask

    task automatic service_page_request(output bit did_work);
        int target;
        int base;
        int count;
        bit store;
        begin
            did_work = 1'b0;
            if (!page_request_word[31]) begin
                return;
            end

            store = page_request_word[30];
            target = page_request_word[27:24];
            base = page_request_word[23:0];

            if (store && (target == T_GELU_PAGE)) begin
                service_gelu_store(base);
                did_work = 1'b1;
            end
            else if (!store && (target == T_GELU_PAGE)) begin
                count = PAGE_WORDS;
                if ((base + count) > GELU_DDR_WORDS) count = GELU_DDR_WORDS - base;
                $display("[TB] GELU load page base=%0d count=%0d at cycle=%0d", base, count, cycle_count);
                load_target_from_source(T_GELU_PAGE, base, count, base, "gelu_page");
                did_work = 1'b1;
            end
            else if (!store && (target == T_IMAGE)) begin
                count = PAGE_WORDS;
                if ((base + count) > IMG_WORDS) count = IMG_WORDS - base;
                $display("[TB] IMAGE load page base=%0d count=%0d at cycle=%0d", base, count, cycle_count);
                load_target_from_source(T_IMAGE, base, count, base, "image_page");
                did_work = 1'b1;
            end
            else if (!store && (target == T_POS)) begin
                count = PAGE_WORDS;
                if ((base + count) > POS_WORDS) count = POS_WORDS - base;
                $display("[TB] POS load page base=%0d count=%0d at cycle=%0d", base, count, cycle_count);
                load_target_from_source(T_POS, base, count, base, "pos_page");
                did_work = 1'b1;
            end
            else begin
                $fatal(1, "[FAIL] unsupported page request word=0x%08x target=%0d store=%0d", page_request_word, target, store);
            end
        end
    endtask

    task automatic service_tile_request(output bit did_work);
        int patch_w_tile;
        int patch_b_tile;
        int trans_w_tile;
        int trans_b_tile;
        begin
            did_work = 1'b0;
            patch_w_tile = patch_request_word[15:0];
            patch_b_tile = patch_request_word[31:16];
            trans_w_tile = trans_request_word[15:0];
            trans_b_tile = trans_request_word[31:16];

            if (status_word[STATUS_PATCH_W_MISS]) begin
                if ((patch_w_tile < 0) || ((patch_w_tile * WEIGHT_TILE_WORDS) >= PATCH_WEIGHT_WORDS)) begin
                    $fatal(1, "[FAIL] patch weight tile id out of range: %0d", patch_w_tile);
                end
                if (verbose_tile) $display("[TB] patch weight tile=%0d at cycle=%0d", patch_w_tile, cycle_count);
                load_target_from_source(T_PW_TILE, patch_w_tile, WEIGHT_TILE_WORDS,
                                        patch_w_tile * WEIGHT_TILE_WORDS, "patch_weight_tile");
                did_work = 1'b1;
            end
            else if (status_word[STATUS_PATCH_B_MISS]) begin
                if ((patch_b_tile < 0) || ((patch_b_tile * BIAS_TILE_WORDS) >= PATCH_BIAS_WORDS)) begin
                    $fatal(1, "[FAIL] patch bias tile id out of range: %0d", patch_b_tile);
                end
                if (verbose_tile) $display("[TB] patch bias tile=%0d at cycle=%0d", patch_b_tile, cycle_count);
                load_target_from_source(T_PB_TILE, patch_b_tile, BIAS_TILE_WORDS,
                                        patch_b_tile * BIAS_TILE_WORDS, "patch_bias_tile");
                did_work = 1'b1;
            end
            else if (status_word[STATUS_TRANS_W_MISS]) begin
                if ((trans_w_tile < 0) || ((trans_w_tile * WEIGHT_TILE_WORDS) >= TRANS_WEIGHT_WORDS)) begin
                    $fatal(1, "[FAIL] transformer weight tile id out of range: %0d", trans_w_tile);
                end
                if (verbose_tile) $display("[TB] transformer weight tile=%0d at cycle=%0d", trans_w_tile, cycle_count);
                load_target_from_source(T_TW_TILE, trans_w_tile, WEIGHT_TILE_WORDS,
                                        trans_w_tile * WEIGHT_TILE_WORDS, "trans_weight_tile");
                did_work = 1'b1;
            end
            else if (status_word[STATUS_TRANS_B_MISS]) begin
                if ((trans_b_tile < 0) || ((trans_b_tile * BIAS_TILE_WORDS) >= TRANS_BIAS_WORDS)) begin
                    $fatal(1, "[FAIL] transformer bias tile id out of range: %0d", trans_b_tile);
                end
                if (verbose_tile) $display("[TB] transformer bias tile=%0d at cycle=%0d", trans_b_tile, cycle_count);
                load_target_from_source(T_TB_TILE, trans_b_tile, BIAS_TILE_WORDS,
                                        trans_b_tile * BIAS_TILE_WORDS, "trans_bias_tile");
                did_work = 1'b1;
            end
        end
    endtask

    task automatic service_once(output bit did_work);
        bit page_did;
        bit tile_did;
        begin
            did_work = 1'b0;
            service_page_request(page_did);
            if (page_did) begin
                did_work = 1'b1;
                return;
            end
            service_tile_request(tile_did);
            did_work = tile_did;
        end
    endtask

    task automatic print_progress;
        begin
            $display("[progress cycle=%0d] busy=%0b done=%0b stage=%0d phase=%0d patch_req=0x%08x trans_req=0x%08x page_req=0x%08x status=0x%08x debug=0x%08x",
                     cycle_count, status_word[STATUS_BUSY], status_word[STATUS_DONE_STICKY],
                     stage_id, stage_phase, patch_request_word, trans_request_word,
                     page_request_word, status_word, debug_word);
        end
    endtask

    task automatic compare_xout;
        int i;
        int lane;
        int word_mismatch;
        int byte_mismatch;
        int first_mismatch_printed;
        int signed diff;
        int abs_diff;
        int max_abs;
        longint unsigned sum_abs;
        logic [31:0] rtl_word;
        logic [31:0] gold_word;
        int rtl_b;
        int gold_b;
        real exact_percent;
        real mean_abs;
        real bram_bw;
        begin
            word_mismatch = 0;
            byte_mismatch = 0;
            first_mismatch_printed = 0;
            max_abs = 0;
            sum_abs = 0;

            for (i = 0; i < X_WORDS; i = i + 1) begin
                read_host_word(T_X_BUF, i, rtl_word);
                gold_word = x_out_gold_mem[i];
                if (rtl_word !== gold_word) begin
                    word_mismatch = word_mismatch + 1;
                end
                for (lane = 0; lane < 4; lane = lane + 1) begin
                    rtl_b = (rtl_word >> (lane * 8)) & 8'hff;
                    gold_b = (gold_word >> (lane * 8)) & 8'hff;
                    diff = rtl_b - gold_b;
                    abs_diff = (diff < 0) ? -diff : diff;
                    sum_abs = sum_abs + abs_diff;
                    if (abs_diff > max_abs) max_abs = abs_diff;
                    if (abs_diff != 0) begin
                        byte_mismatch = byte_mismatch + 1;
                        if (first_mismatch_printed < 12) begin
                            $display("[mismatch] byte_index=%0d word=%0d lane=%0d rtl=%0d golden=%0d diff=%0d",
                                     i * 4 + lane, i, lane, rtl_b, gold_b, diff);
                            first_mismatch_printed = first_mismatch_printed + 1;
                        end
                    end
                end
            end

            exact_percent = 100.0 * $itor(X_WORDS * 4 - byte_mismatch) / $itor(X_WORDS * 4);
            mean_abs = $itor(sum_abs) / $itor(X_WORDS * 4);
            bram_bw = (bram_active_cycles == 0) ? 0.0 :
                      $itor((bram_rd_words + bram_wr_words) * 4) / $itor(bram_active_cycles);

            $display("============================================================");
            $display("[RESULT] Full-image one-block RTL simulation finished.");
            $display("[RESULT] cycles=%0d MACs=%0d", cycle_count, mac_count);
            $display("[RESULT] BRAM rd_words=%0d wr_words=%0d active_cycles=%0d BW=%0.3f B/cycle",
                     bram_rd_words, bram_wr_words, bram_active_cycles, bram_bw);
            $display("[RESULT] pingpong wait=%0d load=%0d overlap=%0d",
                     pingpong_wait_cycles, pingpong_load_cycles, pingpong_overlap_cycles);
            $display("[RESULT] x_out word_mismatch=%0d/%0d byte_mismatch=%0d/%0d exact=%0.3f%% max_abs=%0d mean_abs=%0.4f",
                     word_mismatch, X_WORDS, byte_mismatch, X_WORDS * 4, exact_percent, max_abs, mean_abs);
            $display("============================================================");

            if (byte_mismatch == 0) begin
                $display("[PASS] Full-image x_out matches x_out_packed.hex exactly.");
            end
            else begin
                $fatal(1, "[FAIL] Full-image x_out mismatch. See first mismatch lines above.");
            end
        end
    endtask

    initial begin
        if (!$value$plusargs("VEC_DIR=%s", vec_dir)) begin
            vec_dir = "generated_vectors/case_vit_real_model";
        end

        verbose_tile = $test$plusargs("VERBOSE_TILE");
        $display("[TB] Loading vectors from %s", vec_dir);
        if (verbose_tile) $display("[TB] VERBOSE_TILE enabled.");
        $readmemh({vec_dir, "/image.hex"}, image_mem);
        $readmemh({vec_dir, "/pos.hex"}, pos_mem);
        $readmemh({vec_dir, "/cls.hex"}, cls_mem);
        $readmemh({vec_dir, "/gamma.hex"}, gamma_mem);
        $readmemh({vec_dir, "/patch_weight.hex"}, patch_weight_mem);
        $readmemh({vec_dir, "/patch_bias.hex"}, patch_bias_mem);
        $readmemh({vec_dir, "/transformer_weight.hex"}, trans_weight_mem);
        $readmemh({vec_dir, "/transformer_bias.hex"}, trans_bias_mem);
        $readmemh({vec_dir, "/x_out_packed.hex"}, x_out_gold_mem);

        for (int i = 0; i < GELU_DDR_WORDS; i = i + 1) begin
            gelu_ddr_mem[i] = 32'd0;
        end

        clk = 1'b0;
        rst_n = 1'b0;
        start = 1'b0;
        clear_done = 1'b0;
        transformer_only = 1'b0;
        patch_requant_shift = 6'd0;
        input_wr_valid = 1'b0;
        input_wr_addr = 4'd0;
        input_wr_data = 32'd0;
        input_wr_strb = 4'd0;
        output_host_en = 1'b0;
        output_host_target = 4'd0;
        output_host_addr = 17'd0;
        stage_checkpoint_enable = 1'b0;
        stage_checkpoint_resume = 1'b0;
        gelu_store_done_i = 1'b0;

        repeat (10) @(posedge clk);
        rst_n = 1'b1;
        repeat (10) @(posedge clk);

        $display("[TB] Preloading CLS and gamma through ViT_InputLoadFSM.");
        load_target_from_source(T_CLS, 0, CLS_WORDS, 0, "cls");
        load_target_from_source(T_GAMMA, 0, GAMMA_WORDS, 0, "gamma");

        @(posedge clk);
        start <= 1'b1;
        @(posedge clk);
        start <= 1'b0;

        $display("[TB] Started full-image one-block run.");
        while (!done_sticky) begin
            bit did_work;
            service_once(did_work);
            if (!did_work) begin
                @(posedge clk);
            end

            if (input_error || status_word[STATUS_INPUT_ERROR]) begin
                $fatal(1, "[FAIL] input_error asserted status=0x%08x active_target=%0d count=%0d",
                       status_word, input_active_target, input_word_count);
            end

            if ((cycle_count != 0) && ((cycle_count % PROGRESS_CYCLES) == 0)) begin
                print_progress();
            end

            if (stage_done_pulse) begin
                $display("[TB] stage done id=%0d phase=%0d at cycle=%0d", stage_id, stage_phase, cycle_count);
            end

            if (cycle_count >= MAX_CYCLES) begin
                print_progress();
                $fatal(1, "[FAIL] timeout before done_sticky. Increase MAX_CYCLES if the design is still making progress.");
            end
        end

        $display("[TB] done_sticky asserted at cycle=%0d. Comparing final X_out.", cycle_count);
        compare_xout();
        $finish;
    end

endmodule
