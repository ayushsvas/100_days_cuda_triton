//CUDA CODE FOR EXTRACTING PARTICLE LOCATION AND SIZES FROM HOLOGRAMS VIA ANGULAR SPECTRUM METHOD

// 1. Reconstruct a chunk of z slices by ASM kernel (put the Huygen-Fresnel kernel in constant memory or texture memory(?)) 
// 2. Make tiles in xy -> chunks_z, H, W -> chunks_z, num_tiles, tile_x, tile_y -> num_tiles, chunk_z, tile_x, tile_y
// 3. Bring the 3d tile (1, chunk_z, tile_x, tile_y) to the shared memory
// 4.  