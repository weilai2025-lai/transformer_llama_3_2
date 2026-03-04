module fft
#(
    parameter DATA_WIDTH = 32,
    parameter N          = 16,
    parameter NUM_STAGES = 4,
    parameter BITS       = 14
)(
    // Global I/O
    input  logic        clock,
    input  logic        reset,

    // Input REAL FIFO interface (read side)
    input  logic        in_real_empty,
    input  logic [DATA_WIDTH-1:0] in_real_dout,
    output logic        in_real_rd_en,

    // Input IMAG FIFO interface (read side)
    input  logic        in_imag_empty,
    input  logic [DATA_WIDTH-1:0] in_imag_dout,
    output logic        in_imag_rd_en,

    // Output REAL FIFO interface (write side)
    input  logic        out_real_full,
    output logic        out_real_wr_en,
    output logic [DATA_WIDTH-1:0] out_real_din,

    // Output IMAG FIFO interface (write side)
    input  logic        out_imag_full,
    output logic        out_imag_wr_en,
    output logic [DATA_WIDTH-1:0] out_imag_din
);

    //==========================================================================
    // FSM Definition (4-stage butterfly pipeline)
    //   COMPUTE_RD:  read operands from memory
    //   COMPUTE_MUL: 32x32 multiply
    //   COMPUTE_DQ:  dequantize + compute v
    //   COMPUTE_WB:  butterfly add/sub + writeback
    //==========================================================================
    typedef enum logic [2:0] { IDLE, LOAD, COMPUTE_RD, COMPUTE_MUL, COMPUTE_DQ, COMPUTE_WB, WRITE } state_t;
    state_t state_reg, state_next;

    //==========================================================================
    // Bit-reversal table for N=16
    //==========================================================================
    localparam logic [3:0] BIT_REV [0:15] = '{
        4'd0,  4'd8,  4'd4,  4'd12,
        4'd2,  4'd10, 4'd6,  4'd14,
        4'd1,  4'd9,  4'd5,  4'd13,
        4'd3,  4'd11, 4'd7,  4'd15
    };

    //==========================================================================
    // Twiddle factor table: W_16^k, quantized Q17.14
    //==========================================================================
    localparam signed [DATA_WIDTH-1:0] TW_REAL [0:7] = '{
        32'sh00004000, 32'sh00003B20, 32'sh00002D41, 32'sh0000187D,
        32'sh00000000, 32'shFFFFE783, 32'shFFFFD2BF, 32'shFFFFC4E0
    };

    localparam signed [DATA_WIDTH-1:0] TW_IMAG [0:7] = '{
        32'sh00000000, 32'shFFFFE783, 32'shFFFFD2BF, 32'shFFFFC4E0,
        32'shFFFFC000, 32'shFFFFC4E0, 32'shFFFFD2BF, 32'shFFFFE783
    };

    //==========================================================================
    // Counters
    //==========================================================================
    logic [3:0] cnt_reg,   cnt_next;
    logic [1:0] stage_reg, stage_next;
    logic [2:0] bfly_reg,  bfly_next;

    //==========================================================================
    // Working memory: 16 complex samples
    //==========================================================================
    logic signed [DATA_WIDTH-1:0] mem_real_reg [0:N-1];
    logic signed [DATA_WIDTH-1:0] mem_real_next [0:N-1];
    logic signed [DATA_WIDTH-1:0] mem_imag_reg [0:N-1];
    logic signed [DATA_WIDTH-1:0] mem_imag_next [0:N-1];

    //==========================================================================
    // Pipeline stage 1: COMPUTE_RD → COMPUTE_MUL
    //==========================================================================
    logic signed [DATA_WIDTH-1:0] in1_real_reg, in1_real_next;
    logic signed [DATA_WIDTH-1:0] in1_imag_reg, in1_imag_next;
    logic signed [DATA_WIDTH-1:0] in2_real_reg, in2_real_next;
    logic signed [DATA_WIDTH-1:0] in2_imag_reg, in2_imag_next;
    logic signed [DATA_WIDTH-1:0] w_real_reg, w_real_next;
    logic signed [DATA_WIDTH-1:0] w_imag_reg, w_imag_next;
    logic [3:0] wb_addr1_reg, wb_addr1_next;
    logic [3:0] wb_addr2_reg, wb_addr2_next;

    //==========================================================================
    // Pipeline stage 2: COMPUTE_MUL → COMPUTE_DQ
    //==========================================================================
    logic signed [63:0] prod_wr_i2r_reg, prod_wr_i2r_next;
    logic signed [63:0] prod_wi_i2i_reg, prod_wi_i2i_next;
    logic signed [63:0] prod_wr_i2i_reg, prod_wr_i2i_next;
    logic signed [63:0] prod_wi_i2r_reg, prod_wi_i2r_next;
    logic signed [DATA_WIDTH-1:0] in1_real_p2_reg, in1_real_p2_next;
    logic signed [DATA_WIDTH-1:0] in1_imag_p2_reg, in1_imag_p2_next;
    logic [3:0] wb_addr1_p2_reg, wb_addr1_p2_next;
    logic [3:0] wb_addr2_p2_reg, wb_addr2_p2_next;

    //==========================================================================
    // Pipeline stage 3: COMPUTE_DQ → COMPUTE_WB
    //==========================================================================
    logic signed [DATA_WIDTH-1:0] v_real_reg, v_real_next;
    logic signed [DATA_WIDTH-1:0] v_imag_reg, v_imag_next;
    logic signed [DATA_WIDTH-1:0] in1_real_p3_reg, in1_real_p3_next;
    logic signed [DATA_WIDTH-1:0] in1_imag_p3_reg, in1_imag_p3_next;
    logic [3:0] wb_addr1_p3_reg, wb_addr1_p3_next;
    logic [3:0] wb_addr2_p3_reg, wb_addr2_p3_next;

    //==========================================================================
    // Butterfly address computation (combinational)
    //==========================================================================
    logic [3:0] bfly_addr1, bfly_addr2;
    logic [2:0] tw_j;
    logic [2:0] tw_idx;

    assign tw_idx = tw_j << (2'd3 - stage_reg);

    //==========================================================================
    // COMPUTE_MUL wires: multiply (on registered operands from stage 1)
    //==========================================================================
    logic signed [63:0] prod_wr_i2r_w, prod_wi_i2i_w;
    logic signed [63:0] prod_wr_i2i_w, prod_wi_i2r_w;

    assign prod_wr_i2r_w = w_real_reg * in2_real_reg;
    assign prod_wi_i2i_w = w_imag_reg * in2_imag_reg;
    assign prod_wr_i2i_w = w_real_reg * in2_imag_reg;
    assign prod_wi_i2r_w = w_imag_reg * in2_real_reg;

    //==========================================================================
    // COMPUTE_DQ wires: dequantize (on registered products from stage 2)
    //==========================================================================
    function automatic logic signed [31:0] dequantize(input logic signed [63:0] val);
        logic signed [63:0] rounded;
        rounded = val + 64'sd8192;
        if (rounded >= 0)
            return rounded[31+BITS:BITS];
        else
            if (rounded[BITS-1:0] != '0)
                return (rounded >>> BITS) + 32'sd1;
            else
                return rounded >>> BITS;
    endfunction

    logic signed [DATA_WIDTH-1:0] v_real_w, v_imag_w;

    assign v_real_w = dequantize(prod_wr_i2r_reg) - dequantize(prod_wi_i2i_reg);
    assign v_imag_w = dequantize(prod_wr_i2i_reg) + dequantize(prod_wi_i2r_reg);

    //==========================================================================
    // COMPUTE_WB wires: butterfly add/sub (on registered v from stage 3)
    //==========================================================================
    logic signed [DATA_WIDTH-1:0] out1_real, out1_imag;
    logic signed [DATA_WIDTH-1:0] out2_real, out2_imag;

    assign out1_real = in1_real_p3_reg + v_real_reg;
    assign out1_imag = in1_imag_p3_reg + v_imag_reg;
    assign out2_real = in1_real_p3_reg - v_real_reg;
    assign out2_imag = in1_imag_p3_reg - v_imag_reg;

    //==========================================================================
    // Sequential Process: register updates only
    //==========================================================================
    always_ff @(posedge clock or posedge reset) begin
        if (reset) begin
            state_reg        <= IDLE;
            cnt_reg          <= '0;
            stage_reg        <= '0;
            bfly_reg         <= '0;
            // Stage 1
            in1_real_reg     <= '0;
            in1_imag_reg     <= '0;
            in2_real_reg     <= '0;
            in2_imag_reg     <= '0;
            w_real_reg       <= '0;
            w_imag_reg       <= '0;
            wb_addr1_reg     <= '0;
            wb_addr2_reg     <= '0;
            // Stage 2
            prod_wr_i2r_reg  <= '0;
            prod_wi_i2i_reg  <= '0;
            prod_wr_i2i_reg  <= '0;
            prod_wi_i2r_reg  <= '0;
            in1_real_p2_reg  <= '0;
            in1_imag_p2_reg  <= '0;
            wb_addr1_p2_reg  <= '0;
            wb_addr2_p2_reg  <= '0;
            // Stage 3
            v_real_reg       <= '0;
            v_imag_reg       <= '0;
            in1_real_p3_reg  <= '0;
            in1_imag_p3_reg  <= '0;
            wb_addr1_p3_reg  <= '0;
            wb_addr2_p3_reg  <= '0;
            // Memory
            for (int i = 0; i < N; i++) begin
                mem_real_reg[i] <= '0;
                mem_imag_reg[i] <= '0;
            end
        end
        else begin
            state_reg        <= state_next;
            cnt_reg          <= cnt_next;
            stage_reg        <= stage_next;
            bfly_reg         <= bfly_next;
            // Stage 1
            in1_real_reg     <= in1_real_next;
            in1_imag_reg     <= in1_imag_next;
            in2_real_reg     <= in2_real_next;
            in2_imag_reg     <= in2_imag_next;
            w_real_reg       <= w_real_next;
            w_imag_reg       <= w_imag_next;
            wb_addr1_reg     <= wb_addr1_next;
            wb_addr2_reg     <= wb_addr2_next;
            // Stage 2
            prod_wr_i2r_reg  <= prod_wr_i2r_next;
            prod_wi_i2i_reg  <= prod_wi_i2i_next;
            prod_wr_i2i_reg  <= prod_wr_i2i_next;
            prod_wi_i2r_reg  <= prod_wi_i2r_next;
            in1_real_p2_reg  <= in1_real_p2_next;
            in1_imag_p2_reg  <= in1_imag_p2_next;
            wb_addr1_p2_reg  <= wb_addr1_p2_next;
            wb_addr2_p2_reg  <= wb_addr2_p2_next;
            // Stage 3
            v_real_reg       <= v_real_next;
            v_imag_reg       <= v_imag_next;
            in1_real_p3_reg  <= in1_real_p3_next;
            in1_imag_p3_reg  <= in1_imag_p3_next;
            wb_addr1_p3_reg  <= wb_addr1_p3_next;
            wb_addr2_p3_reg  <= wb_addr2_p3_next;
            // Memory
            for (int i = 0; i < N; i++) begin
                mem_real_reg[i] <= mem_real_next[i];
                mem_imag_reg[i] <= mem_imag_next[i];
            end
        end
    end

    //==========================================================================
    // Combinational Process: FSM + data path control
    //==========================================================================
    always_comb begin
        // Default: hold all registers
        state_next        = state_reg;
        cnt_next          = cnt_reg;
        stage_next        = stage_reg;
        bfly_next         = bfly_reg;
        // Stage 1
        in1_real_next     = in1_real_reg;
        in1_imag_next     = in1_imag_reg;
        in2_real_next     = in2_real_reg;
        in2_imag_next     = in2_imag_reg;
        w_real_next       = w_real_reg;
        w_imag_next       = w_imag_reg;
        wb_addr1_next     = wb_addr1_reg;
        wb_addr2_next     = wb_addr2_reg;
        // Stage 2
        prod_wr_i2r_next  = prod_wr_i2r_reg;
        prod_wi_i2i_next  = prod_wi_i2i_reg;
        prod_wr_i2i_next  = prod_wr_i2i_reg;
        prod_wi_i2r_next  = prod_wi_i2r_reg;
        in1_real_p2_next  = in1_real_p2_reg;
        in1_imag_p2_next  = in1_imag_p2_reg;
        wb_addr1_p2_next  = wb_addr1_p2_reg;
        wb_addr2_p2_next  = wb_addr2_p2_reg;
        // Stage 3
        v_real_next       = v_real_reg;
        v_imag_next       = v_imag_reg;
        in1_real_p3_next  = in1_real_p3_reg;
        in1_imag_p3_next  = in1_imag_p3_reg;
        wb_addr1_p3_next  = wb_addr1_p3_reg;
        wb_addr2_p3_next  = wb_addr2_p3_reg;
        // Memory
        for (int i = 0; i < N; i++) begin
            mem_real_next[i] = mem_real_reg[i];
            mem_imag_next[i] = mem_imag_reg[i];
        end

        // Default outputs
        in_real_rd_en  = 1'b0;
        in_imag_rd_en  = 1'b0;
        out_real_wr_en = 1'b0;
        out_imag_wr_en = 1'b0;
        out_real_din   = '0;
        out_imag_din   = '0;

        // Default butterfly address
        bfly_addr1 = 4'd0;
        bfly_addr2 = 4'd0;
        tw_j       = 3'd0;

        // Butterfly address mapping
        case (stage_reg)
            2'd0: begin
                bfly_addr1 = {bfly_reg, 1'b0};
                bfly_addr2 = {bfly_reg, 1'b1};
                tw_j       = 3'd0;
            end
            2'd1: begin
                bfly_addr1 = {bfly_reg[2:1], 1'b0, bfly_reg[0]};
                bfly_addr2 = {bfly_reg[2:1], 1'b1, bfly_reg[0]};
                tw_j       = {2'd0, bfly_reg[0]};
            end
            2'd2: begin
                bfly_addr1 = {bfly_reg[2], 1'b0, bfly_reg[1:0]};
                bfly_addr2 = {bfly_reg[2], 1'b1, bfly_reg[1:0]};
                tw_j       = {1'd0, bfly_reg[1:0]};
            end
            2'd3: begin
                bfly_addr1 = {1'b0, bfly_reg};
                bfly_addr2 = {1'b1, bfly_reg};
                tw_j       = bfly_reg;
            end
            default: ;
        endcase

        case (state_reg)
            //------------------------------------------------------------------
            IDLE: begin
                if (!in_real_empty && !in_imag_empty) begin
                    cnt_next   = '0;
                    state_next = LOAD;
                end
            end

            //------------------------------------------------------------------
            LOAD: begin
                if (!in_real_empty && !in_imag_empty) begin
                    in_real_rd_en = 1'b1;
                    in_imag_rd_en = 1'b1;

                    mem_real_next[BIT_REV[cnt_reg]] = signed'(in_real_dout);
                    mem_imag_next[BIT_REV[cnt_reg]] = signed'(in_imag_dout);

                    if (cnt_reg == 4'd15) begin
                        stage_next = '0;
                        bfly_next  = '0;
                        state_next = COMPUTE_RD;
                    end
                    else begin
                        cnt_next = cnt_reg + 4'd1;
                    end
                end
            end

            //------------------------------------------------------------------
            // COMPUTE_RD: Read operands from memory → register
            //------------------------------------------------------------------
            COMPUTE_RD: begin
                in1_real_next = mem_real_reg[bfly_addr1];
                in1_imag_next = mem_imag_reg[bfly_addr1];
                in2_real_next = mem_real_reg[bfly_addr2];
                in2_imag_next = mem_imag_reg[bfly_addr2];
                w_real_next   = TW_REAL[tw_idx];
                w_imag_next   = TW_IMAG[tw_idx];
                wb_addr1_next = bfly_addr1;
                wb_addr2_next = bfly_addr2;
                state_next    = COMPUTE_MUL;
            end

            //------------------------------------------------------------------
            // COMPUTE_MUL: 32×32 multiply → register 64-bit products
            //------------------------------------------------------------------
            COMPUTE_MUL: begin
                prod_wr_i2r_next = prod_wr_i2r_w;
                prod_wi_i2i_next = prod_wi_i2i_w;
                prod_wr_i2i_next = prod_wr_i2i_w;
                prod_wi_i2r_next = prod_wi_i2r_w;
                in1_real_p2_next = in1_real_reg;
                in1_imag_p2_next = in1_imag_reg;
                wb_addr1_p2_next = wb_addr1_reg;
                wb_addr2_p2_next = wb_addr2_reg;
                state_next       = COMPUTE_DQ;
            end

            //------------------------------------------------------------------
            // COMPUTE_DQ: Dequantize products → compute v_real, v_imag
            //------------------------------------------------------------------
            COMPUTE_DQ: begin
                v_real_next      = v_real_w;
                v_imag_next      = v_imag_w;
                in1_real_p3_next = in1_real_p2_reg;
                in1_imag_p3_next = in1_imag_p2_reg;
                wb_addr1_p3_next = wb_addr1_p2_reg;
                wb_addr2_p3_next = wb_addr2_p2_reg;
                state_next       = COMPUTE_WB;
            end

            //------------------------------------------------------------------
            // COMPUTE_WB: Butterfly add/sub + writeback to memory
            //------------------------------------------------------------------
            COMPUTE_WB: begin
                mem_real_next[wb_addr1_p3_reg] = out1_real;
                mem_imag_next[wb_addr1_p3_reg] = out1_imag;
                mem_real_next[wb_addr2_p3_reg] = out2_real;
                mem_imag_next[wb_addr2_p3_reg] = out2_imag;

                if (bfly_reg == 3'd7) begin
                    bfly_next = '0;
                    if (stage_reg == 2'd3) begin
                        cnt_next   = '0;
                        state_next = WRITE;
                    end
                    else begin
                        stage_next = stage_reg + 2'd1;
                        state_next = COMPUTE_RD;
                    end
                end
                else begin
                    bfly_next  = bfly_reg + 3'd1;
                    state_next = COMPUTE_RD;
                end
            end

            //------------------------------------------------------------------
            WRITE: begin
                if (!out_real_full && !out_imag_full) begin
                    out_real_wr_en = 1'b1;
                    out_imag_wr_en = 1'b1;
                    out_real_din   = mem_real_reg[cnt_reg];
                    out_imag_din   = mem_imag_reg[cnt_reg];

                    if (cnt_reg == 4'd15) begin
                        state_next = IDLE;
                    end
                    else begin
                        cnt_next = cnt_reg + 4'd1;
                    end
                end
            end

            default: state_next = IDLE;
        endcase
    end

endmodule
