module Multilayer_Perceptron(input clk, input reset, input [15:0] address, input [31:0] din, output reg [31:0] dout, input write, input read);
reg [15:0] control_reg;
reg [7:0] N;
reg [7:0] M;
reg [7:0] H;
wire [7:0]mem_dat_data_a; 
wire [15:0] status_out;
reg mem_w_wr, mem_f_wr;
wire mem_dat_en, meme_w_en, mem_F_en;
wire [9:0] mem_dat_add_rd, mem_dat_add_wr;
reg [9:0] mem_dat_add_wr_m;
wire [7:0] mem_dat_data, mem_dat_data_f; 
reg [7:0] mem_dat_data_f_m;
wire mem_dat_wr;
reg mem_dat_wr_m;
wire [11:0] mem_w_address, mem_w_address2, mem_F_address;
reg [11:0] mem_F_address_m, mem_w_address_m;
wire [15:0] mem_w_data, mem_w_data2;
reg [7:0] mem_f_data_w;
reg [15:0] mem_w_data_w;

mlp_core core (clk, reset, control_reg, N, M, H, mem_dat_en, mem_dat_add_rd, mem_dat_data, 
mem_dat_add_wr, mem_dat_wr, mem_w_address, mem_w_address2, meme_w_en, mem_w_data, mem_w_data2,
mem_F_address, mem_F_en, status_out);

always @(posedge clk)
  if (write)
    case (address)
      16'h0004: control_reg<=din [15:0];
      16'h0010: N<=din;
      16'h0014: M<=din;
      16'h0018: H<=din;
    endcase


always @(*) begin
  case (address[15:12]) 
    4'h0: 
      case (address [7:0]) 
        8'h04: dout=control_reg;
        8'h08: dout=status_out;
        8'h10: dout=N;
        8'h14: dout=M;
        8'h18: dout=H;
      endcase
    4'h1: dout=mem_dat_data_a;
    4'h4: dout=mem_dat_data_f;
    4'h8: dout=mem_w_data;  
    4'hA: dout=mem_w_data;
  endcase
end

always @(*) begin
  if (address [15:12]==4'h1 && control_reg[0]) begin
    mem_dat_wr_m=write;
    mem_dat_data_f_m=din;
    if (address [11:10]==2'b00)
      mem_dat_add_wr_m={2'b00, address [7:0]};
    else if (address [11:10]==2'b01)
      mem_dat_add_wr_m={2'b01, address [7:0]};
    else
      mem_dat_add_wr_m={2'b10, address [7:0]};
  end
  else begin
    mem_dat_wr_m=mem_dat_wr;
    mem_dat_data_f_m=mem_dat_data_f;
    mem_dat_add_wr_m=mem_dat_add_wr;
  end
  if (address [15:12]==4'h4 && control_reg[0]) begin    
    mem_f_wr=write;
    mem_f_data_w=din;
    mem_F_address_m=address [11:0];
  end
  else begin
    mem_f_wr=1'b0;
    mem_f_data_w=8'b0;
    mem_F_address_m=mem_F_address;
  end
  if (address [15]==1'b1 && control_reg[0]) begin
    mem_w_wr=write;
    mem_w_data_w=din;
    if (address [14:12]==0)
      mem_w_address_m={1'b0, address[10:0]};
    else
      mem_w_address_m={1'b1, address[10:0]};
  end
  else begin
    mem_w_wr=0;
    mem_w_data_w=12'b0;
    mem_w_address_m=mem_w_address;
  end
end

endmodule

module mlp_core (clk, reset, control, N, M, H, mem_dat_en, mem_dat_add_rd, mem_dat_data, 
                mem_dat_add_wr, mem_dat_wr, mem_w_address, mem_w_address2, meme_w_en, mem_w_data, mem_w_data2, 
                mem_F_address, mem_F_en, status_r);
input clk, reset;
input [15:0] control; 
input [7:0] N, M, H;
output mem_dat_en, meme_w_en, mem_F_en;
output reg [9:0] mem_dat_add_rd, mem_dat_add_wr;
reg [9:0] mem_dat_add_wr2; 
output reg [11:0] mem_w_address, mem_F_address, mem_w_address2; 
reg [11:0] mem_F_address2;
input [7:0] mem_dat_data;
output reg mem_dat_wr;
input [15:0] mem_w_data, mem_w_data2;
output reg [15:0] status_r;

parameter data_in_m=10'h000;
parameter data_a_m=10'h100;
parameter data_out_m=10'h200;
parameter data_a_m2=10'h101;
parameter data_out_m2=10'h201;
parameter iw_m=12'h000;
parameter il_m=12'h800;

reg signed [7:0] data;
reg signed [15:0] weight, weight2;
reg signed [23:0] mult, mult2;
reg signed [31:0] suma, suma2;
reg [7:0] ctrl_0, ctrl_1, ctrl_2, ctrl_3, ctrl_4, ctrl_5;
reg [7:0] cont; reg [7:0] cont2;

assign mem_dat_en=1;
assign meme_w_en=1;
assign mem_F_en=1;
 

always @ (posedge clk) begin
  if (reset || control[2] || status_r[0]) begin
    ctrl_0<=0; ctrl_1<=0; ctrl_2<=0; cont<=1; 
    ctrl_3<=0; ctrl_4<=0; ctrl_5<=0; cont2<=2;
    mem_dat_add_rd<=data_in_m;
    mem_dat_add_wr<=data_a_m;
    mem_dat_add_wr2<=data_a_m2;
    mem_w_address<=12'h000;
    mem_w_address2<=12'h000+N;
  end
  
  if (reset || control[2]) begin
    status_r<=0;
  end
  else begin

    if (status_r[0]==1 && control[1]==0)
      status_r[0]=0;
      
    else if (status_r[0]==0 && control[1]==1) begin
      ctrl_0[0]<=1; ctrl_1<=ctrl_0; ctrl_2<=ctrl_1;
      ctrl_3<=ctrl_2; ctrl_5<=ctrl_4;
      data<=mem_dat_data;
      weight<=mem_w_data;
      mult<=data*weight;
      weight2<=mem_w_data2;
      mult2<=data*weight2;

      if (cont==N && cont2>=M && ctrl_0[4]==0) begin
        mem_dat_add_rd<=data_a_m;
        mem_w_address<=il_m;
        mem_w_address2<=il_m+M;
        ctrl_0[3:1]<=3'b010;
        ctrl_0[4]<=1;
        if (cont2>M)
          ctrl_0[5]<=1;
        else
          ctrl_0[5]<=0;
        cont<=1;
        cont2<=2;
      end
      else if (cont==M && cont2>=H && ctrl_0[4]==1) begin
        ctrl_0[3:1]<=3'b100;
        cont<=0;
        if (cont2>H)
          ctrl_0[5]<=1;
        else
          ctrl_0[5]<=0;
      end
      else if (cont==N && ctrl_0[4]==0) begin
        ctrl_0[3:1]<=3'b001;
        mem_dat_add_rd<=data_in_m;
        mem_w_address<=mem_w_address+N+1;
        mem_w_address2<=mem_w_address2+N+1;
        cont<=1;
        cont2<=cont2+2;
        ctrl_0[5]<=0;
      end
      else if (cont==M && ctrl_0[4]==1) begin
        ctrl_0[3:1]<=3'b001;
        mem_dat_add_rd<=data_a_m;
        mem_w_address<=mem_w_address+M+1;
        mem_w_address2<=mem_w_address2+M+1;
        cont<=1;
        cont2<=cont2+2;
        ctrl_0[5]<=0;
      end
      else begin
        mem_dat_add_rd<=mem_dat_add_rd+1;
        mem_w_address<=mem_w_address+1;
        mem_w_address2<=mem_w_address2+1;
        cont<=cont+1;
        ctrl_0[3:1]<=3'b000;
        ctrl_0[5]<=0;
      end

      if (ctrl_3[3:1]!=3'b000) begin
        suma<=mult;
        suma2<=mult2;
      end
      else if (ctrl_2[0]) begin 
        suma<=suma+mult;
        suma2<=suma2+mult2;
      end
      else begin
        suma<=0;
        suma2<=0;
      end
      
      if (ctrl_3[3:1]) begin
        if ((suma[30:17]!=0 && suma[31]==0) || ctrl_3[2])
         mem_F_address<=12'h7FF;
        else if (suma[30:17]!=14'hFFFF && suma[31])
          mem_F_address<=12'h800;
        else
          mem_F_address<={suma[31],suma[16:6]};
          
        if ((suma2[30:17]!=0 && suma2[31]==0) || ctrl_3[2])
           mem_F_address2<=12'h7FF;
        else if (suma2[30:17]!=14'hFFFF && suma2[31])
           mem_F_address2<=12'h800;
        else
           mem_F_address2<={suma2[31],suma2[16:6]};
      end

      if (ctrl_4[3:1]) begin
        mem_dat_wr<=1;
        if (ctrl_4[5] || ctrl_4[6])
          ctrl_4<=ctrl_3;
        else begin
          ctrl_4[6]<=1;
          mem_F_address<=mem_F_address2;
        end
      end
      else begin
        mem_dat_wr<=0;
        ctrl_4<=ctrl_3;
      end
      
      if (ctrl_5[3:1]!=0 && ctrl_5[5]==0 && ctrl_5[6]==0) begin
          mem_dat_add_wr<=mem_dat_add_wr2;
          mem_dat_add_wr2<=mem_dat_add_wr;
      end
      else if (ctrl_5[1]) begin
          mem_dat_add_wr<=mem_dat_add_wr2+2;
          mem_dat_add_wr2<=mem_dat_add_wr+2;
      end
      else if (ctrl_5[2]) begin
          mem_dat_add_wr<=data_out_m;
          mem_dat_add_wr2<=data_out_m2;
      end

      if (ctrl_5[3] && (ctrl_5[5] || ctrl_5[6]))
        status_r[0]<=1;
    end
    else
      mem_w_address2<=12'h000+N;
  end
end
endmodule

