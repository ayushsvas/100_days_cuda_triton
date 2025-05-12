//TYPES OF MEMORIES IN CUDA -- THE VON-NEUMANN PROCESSOR 
//The aim is to maximise the compute to global memory access (CGMA) ratio.
//Global memory is implemented as DRAM which is slow and handles memory bandwidth 
//Jensen provided with shared memory per block, registers per thread, constant memory for all threads (read only)
//Von-Neumann model is "stored program" model (contains a control unit with  PC, IR and processing unit with ALU, Register File)
// There's program counter (PC) which keeps track of the instruction at bay (constains
// the memory address of the next instruction to be executed), fetches it from memory to instruction register (IR).
// This fetching from global memory (which is off chip DRAM), is slow. More the intructions, the slower the program gets
// It's nice to declare operands directly to registers in register file (on chip), no instruction then required to make 
// the operand value available to ALU. Placing operands in registers can improve execution speed.