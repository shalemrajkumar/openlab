__global__ void twoSeconds(float *out)
{
  // calculate block ID 
  int blockId = blockIdx.x + gridDim.x * blockIdx.y + gridDim.x * gridDim.y * blockIdx.z;

  // calculate global thread ID
  int threadId = threadIdx.x + blockDim.x * threadIdx.y + blockDim.x * blockDim.y * threadIdx.z;
  
  int index = blockId * (blockDim.x * blockDim.y * blockDim.z) + threadId;

  float x0 = (float)index;  // stays in register, no memory traffic

  // simulate work for 2 seconds
  float x1 = x0 + 1.0f;
  float x2 = x0 + 2.0f;
  float x3 = x0 + 3.0f;

  for (unsigned int i = 0; i < 795000000U; i++) 
  {
      x0 = x0 * 1.0001f + 0.0001f;
      x1 = x1 * 1.0001f + 0.0001f;
      x2 = x2 * 1.0001f + 0.0001f;
      x3 = x3 * 1.0001f + 0.0001f;
  }
  out[index] = x0 + x1 + x2 + x3; // single write at end
}


int main()
{
  unsigned int b_1 = 14, b_2 = 4, b_3 = 1; // grid dimensions
  unsigned int t_1 = 4, t_2 = 4, t_3 = 2; // block dimensions -> warp
                                          
  unsigned int blocks_per_grid = b_1 * b_2 * b_3; // 32*4*14 = 1792 blocksPerGrid
  unsigned int threads_per_block = t_1 * t_2 * t_3; // 4*4*4 = 64 threadsPerBlock
  
  dim3 blocksPerGrid(b_1, b_2, b_3); // 32 blocks in x, 4 blocks in y, 14 blocks in z
  dim3 threadsPerBlock(t_1, t_2, t_3); // 4 threads in x, 4 threads in y, 4 threads in z
  
  float *out;
  cudaMalloc(&out, sizeof(float) * blocks_per_grid * threads_per_block); // allocate output array on device (only 1792 floats = ~7KB)


  // kernel lauch
  twoSeconds<<<blocksPerGrid, threadsPerBlock>>>(out);
  
  cudaDeviceSynchronize();
  cudaFree(out);

  return 0;
}
