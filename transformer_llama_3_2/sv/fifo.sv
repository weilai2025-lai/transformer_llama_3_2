module fifo
#(
	parameter data_width = 24,
	parameter buffer_size = 1024
)(
	//global i/o
	input logic rst,
	//writing interface
	input logic wr_clk, wr_en,
	input logic [data_width-1:0] din,
	output logic full,
	//reading interface
	input logic rd_clk, rd_en,
	output logic empty,
	output logic [data_width-1:0] dout
);

	//function used to set 0 for all unknown value
	function automatic logic [data_width-1:0] to01(input logic [data_width-1:0] data);
		logic [data_width-1:0] result;
		for(int i = 0; i < $bits(data); i++) begin
			case(data[i])
				0: result[i] = 1'b0;
				1: result[i] = 1'b1;
				default: result[i] = 1'b0;
			endcase
		end
		return result;
	endfunction

	//define unpacked array
	logic [data_width-1:0] buffer[buffer_size-1:0];
	
	//define writing & reading address (register and next signal), use one more bit to verify empty and full status
	localparam address_width = $clog2(buffer_size) + 1;
	logic [address_width-1:0] wr_addr_reg, wr_addr_next;
	logic [address_width-1:0] rd_addr_reg, rd_addr_next;
	
	//define other register and next signal
	logic empty_reg, empty_next;
	logic full_reg, full_next;
	
	//define other signal
	logic move_in;
	
	//Sequential process: writing procedure 
	always_ff @(posedge wr_clk) begin
		if(rst) begin
			full_reg <= 1'b0;
			wr_addr_reg <= '0;
		end
		else begin
			full_reg <= full_next;
			wr_addr_reg <= wr_addr_next;
			if(wr_en && !full_reg) begin
				buffer[wr_addr_reg[address_width-2:0]] <= din;
			end
		end
	end
	
	//Combinational process: writing procedure 
	always_comb begin
		wr_addr_next = (wr_en && !full_reg) ? (wr_addr_reg + 1'b1) : wr_addr_reg;
		full_next = (wr_addr_next[address_width-1] != rd_addr_reg[address_width-1]) &&
		(wr_addr_next[address_width-2:0] == rd_addr_reg[address_width-2:0]);
	end
	
	//Sequential process: reading procedure 
	always_ff @(posedge rd_clk) begin
		if(rst) begin
			empty_reg <= 1'b1;
			rd_addr_reg <= '0;
		end
		else begin
			rd_addr_reg <= rd_addr_next;
			empty_reg <= empty_next;
		end
	end
	
	//Combinational process: reading procedure 
	always_comb begin
		rd_addr_next = (rd_en && !empty_reg) ? rd_addr_reg + 1'b1 : rd_addr_reg;
		empty_next = (rd_addr_next == wr_addr_reg);
	end

	//assign block
	assign dout = to01(buffer[$unsigned(rd_addr_reg[address_width-2:0])]);
	assign empty = empty_reg;
	assign full = full_reg;
endmodule