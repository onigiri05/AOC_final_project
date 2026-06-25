module OutputCaptureFSM #(
    parameter ADDR_WIDTH = 8,
    parameter OUTPUT_WORDS = 256
)(
    input clk,
    input rst_n,

    input start,
    input opsum_valid,
    input [31:0] opsum,

    output reg capture_busy,
    output reg capture_done,
    output reg capture_overflow,
    output reg [ADDR_WIDTH:0] capture_count,

    output reg mem_wr_en,
    output reg [ADDR_WIDTH-1:0] mem_wr_addr,
    output reg [31:0] mem_wr_data,
    output reg [3:0] mem_wr_byte_en
);

localparam [ADDR_WIDTH:0] OUTPUT_WORDS_L = OUTPUT_WORDS;

wire last_word = (capture_count == (OUTPUT_WORDS_L - 1'b1));

always @(posedge clk) begin
    if(!rst_n) begin
        capture_busy <= 1'b0;
        capture_done <= 1'b0;
        capture_overflow <= 1'b0;
        capture_count <= {(ADDR_WIDTH+1){1'b0}};
        mem_wr_en <= 1'b0;
        mem_wr_addr <= {ADDR_WIDTH{1'b0}};
        mem_wr_data <= 32'd0;
        mem_wr_byte_en <= 4'd0;
    end
    else begin
        mem_wr_en <= 1'b0;

        if(start) begin
            capture_busy <= 1'b1;
            capture_done <= 1'b0;
            capture_overflow <= 1'b0;
            capture_count <= {(ADDR_WIDTH+1){1'b0}};
        end
        else if(opsum_valid && capture_busy) begin
            mem_wr_en <= 1'b1;
            mem_wr_addr <= capture_count[ADDR_WIDTH-1:0];
            mem_wr_data <= opsum;
            mem_wr_byte_en <= 4'hf;

            if(last_word) begin
                capture_busy <= 1'b0;
                capture_done <= 1'b1;
                capture_count <= OUTPUT_WORDS_L;
            end
            else begin
                capture_count <= capture_count + 1'b1;
            end
        end
        else if(opsum_valid && capture_done) begin
            capture_overflow <= 1'b1;
        end
    end
end

endmodule
