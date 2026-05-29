#include <iostream>
using namespace std;


__global__ void whoAmI()
{
  // calculate block ID 
  int blockId = blockIdx.x + gridDim.x * blockIdx.y + gridDim.x * gridDim.y * blockIdx.z;

  // calculate global thread ID
  int threadId = threadIdx.x + blockDim.x * threadIdx.y + blockDim.x * blockDim.y * threadIdx.z;

  printf("Hello from thread %d at block %d \n", threadId, blockId);
}


int main()
{
  unsigned int b_1 = 14, b_2 = 4, b_3 = 1; // grid dimensions
  unsigned int t_1 = 4, t_2 = 4, t_3 = 2; // block dimensions -> warp
                                          
  unsigned int blocks_per_grid = b_1 * b_2 * b_3; // 32*4*14 = 1792 blocksPerGrid
  unsigned int threads_per_block = t_1 * t_2 * t_3; // 4*4*4 = 64 threadsPerBlock
  
  dim3 blocksPerGrid(b_1, b_2, b_3); // 32 blocks in x, 4 blocks in y, 14 blocks in z
  dim3 threadsPerBlock(t_1, t_2, t_3); // 4 threads in x, 4 threads in y, 4 threads in z
                                       
  cout << "blocksPerGrid: " << blocks_per_grid << endl; // 1792
  cout << "threadsPerBlock: " << threads_per_block << endl; // 64
  
  // kernel lauch
  whoAmI<<<blocksPerGrid, threadsPerBlock>>>();
  cudaDeviceSynchronize();

  return 0;
}
