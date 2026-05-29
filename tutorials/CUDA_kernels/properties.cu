#include <iostream>
using namespace std;

int main() 
{
  
  int device;
  cudaGetDevice(&device);

  struct cudaDeviceProp props;
  cudaGetDeviceProperties(&props, device);

  cout << "Device id" << device << endl;
  cout << "Device name: " << props.name << endl;

  return 0;
}
