#include <stdio.h>

int main() {

  int nDevices;
  cudaGetDeviceCount(&nDevices);
  
  printf("Number of devices: %d\n", nDevices);
  
  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("Device Number: %d\n", i);
    printf("  Device name: %s\n", prop.name);
    printf("\n");
    printf("  %%%%%%%%%%%%%%%%%%%%%%%%%% Memory info  %%%%%%%%%%%%%%%%%%%%%%%%%%\n");
    printf("\n");
    printf("  Total Global Memory (Mbytes): %f%.1f\n",prop.totalGlobalMem, (float) (prop.totalGlobalMem) / 1024.0 / 1024.0);
    printf("  Shared Memory per Block (Kbytes): %f%.1f\n", prop.sharedMemPerBlock, (float) (prop.sharedMemPerBlock) / 1024.0);
    printf("  Reserved Shared Memory per Block (Kbytes): %f%.1f\n", prop.reservedSharedMemPerBlock, (float) (prop.reservedSharedMemPerBlock) / 1024.0);
    printf("  Registers per Block: %d\n", prop.regsPerBlock);
    printf("  Registers per Multiprocessor: %d\n", prop.regsPerMultiprocessor);
    printf("  Warp Size: %d\n", prop.warpSize);
    printf("  Memory pitch (bytes): %d\n", prop.memPitch);
    printf("  L2 Cache Size (Kbytes): %d\n", prop.l2CacheSize / 1024);
    printf("\n");
    printf(" %%%%%%%%%%%%%%%%%%%%%%%%%%  Hardware limits %%%%%%%%%%%%%%%%%%%%%%%%%% \n");
    printf("\n");
    printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("  Max threads per multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("  Max threads dim: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("  Max grid size: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("\n");
    printf("  %%%%%%%%%%%%%%%%%%%%%%%%%% SMs %%%%%%%%%%%%%%%%%%%%%%%%%% /n");
    printf("\n");
    printf("  Number of multiprocessors: %d\n", prop.multiProcessorCount);
    printf("\n");
    printf("  %%%%%%%%%%%%%%%%%%%%%%%%%% others %%%%%%%%%%%%%%%%%%%%%%%%%% \n");
    printf("\n");
    printf("  Memory Bus Width (bits): %d\n",prop.memoryBusWidth);
    printf("  minor-major: %d-%d\n", prop.minor, prop.major);
    printf("  Concurrent kernels: %s\n", prop.concurrentKernels ? "yes" : "no");
    printf("  Concurrent computation/communication: %s\n\n",prop.concurrentManagedAccess ? "yes" : "no");
  }
}

