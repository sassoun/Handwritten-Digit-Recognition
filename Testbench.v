`timescale 1ns/10ps
`define Default 32'hFFFF_FFFF

module Testbench ();
reg clk;
reg reset;
reg [15:0] address;
reg [31:0] din;
wire [31:0] dout;
reg write;
reg read;
reg signed [7:0] s;

reg signed [7:0] p;
integer file, j;
real f; integer st;

Multilayer_Perceptron mlp (clk, reset, address, din, dout, write, read);

always
begin
clk=1; #5; clk=0; #5;
end

initial begin

  
  reset=1;
  #19;
  reset=0;
  
  //control parameters and memory location
  write=1;
  address=16'h0004;
  din=16'b001;
  #10
  
  //register N
  address=16'h0010;
  din=36;
  #10
  
  //register M
  address=16'h0014;
  din=11;
  #10
  
  //register H
  address=16'h0018;
  din=26;
  #10
  
  //accessing the input va;ues from file
  address=16'h1000;
  file=$fopen("C:/Users/sassoun77/InputWeights.txt", "r"); // the weights that we found from first layer of neural network
  st=$fscanf(file, "%f", f);
  j=0;
  while (j<=255 && st!=`Default) begin
    //p=f/2;
    din=f/2;
    #10;
    st=$fscanf(file, "%f", f);
    j=j+1;
    address=address+1;
  end
  $fclose (file);

  //Hidden Layer weights 
  address=16'h8000;
  file=$fopen("C:/Users/sassoun77/HiddenLayerWeights.txt", "r"); // the weights that we found from last hidden layer of neural network
  st=$fscanf(file, "%f", f);
  j=0;
  while (j<=2048 && st!=`Default) begin
    din<=f*512;#10;
    st=$fscanf(file, "%f", f);
    j=j+1;
    address=address+1;
  end
  $fclose (file);
  
  //Output layer weights
  address=16'hA000;
  file=$fopen(("C:/Users/sassoun77/OutputsWeights.txt", "r"); // the weights that we found from the last layer of neural network
  st=$fscanf(file, "%f", f);
  j=0;
  while (j<=2048 && st!=`Default) begin
    din=f*512;#10;
    st=$fscanf(file, "%f", f);
    j=j+1;
    address=address+1;
  end
  $fclose (file);
  
  //once the memories are written, the access to the memory is deactvated and the processing is initiated through the control register
  address=16'h0004;
  din=16'b010;
  #10;
  write=0;
  
  //keep the state recorded 
   address=16'h0008;
end

//whe bit 1 of the register is set to 1, display the results stored in the data memory
always @(posedge dout[0]) begin
  if (address==16'h0008) begin
    write<=1;
    address<=16'h0004;
    din<=16'b001;
    #10
    write<=0;
    address<=16'h1800;
    j=0;
    while (j<26) begin
      #10;
      s=dout;
      $display ("Output: %d", s);
      j=j+1;
      address<=address+1;
    end
  end
end
  
endmodule
