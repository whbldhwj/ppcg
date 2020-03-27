#include <assert.h>
#include <stdio.h>
#include "xcl2.hpp"
#include <algorithm>
#include <vector>
#include "kernel_kernel.h"

#include "kernel_top_gen.h"
#include "kernel.h"

int main(int argc, char **argv) {
//  data_t A[I][K], B[K][J], C[I][J], C_golden[I][J]; 
  data_t A[I][K], B[J][K], C[I][J], C_golden[I][J];

  {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <XCLBIN File>" << std::endl;
        return EXIT_FAILURE;
    }

    std::string binaryFile = argv[1];
    cl_int err;
    cl::Context context;
    cl::Kernel krnl;
    cl::CommandQueue q;
    // get_xil_devices() is a utility API which will find the xilinx
    // platforms and will return list of devices connected to Xilinx platform
    auto devices = xcl::get_xil_devices();
    // read_binary_file() is a utility API which will load the binaryFile
    // and will return the pointer to file buffer
    auto fileBuf = xcl::read_binary_file(binaryFile);
    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
    int valid_device = 0;
    for (unsigned int i = 0; i < devices.size(); i++) {
        auto device = devices[i];
        // Creating Context and Command Queue for selected Device
        OCL_CHECK(err, context = cl::Context({device}, NULL, NULL, NULL, &err));
        OCL_CHECK(err, q = cl::CommandQueue(context, {device}, CL_QUEUE_PROFILING_ENABLE, &err));
        std::cout << "Trying to program device[" << i
            << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
        OCL_CHECK(err, cl::Program program(context, {device}, bins, NULL, %err));
        if (err != CL_SUCCESS) {
            std::cout << "Failed to program device[" << i << "] with xclbin file!\n";
        } else {
            std::cout << "Device[" << i << "]: program successful!\n";
            OCL_CHECK(err, krnl = cl::Kernel(program, "kernel0", &err));
            valid_device++
            break; // we break because we found a valid device
        }
    }
    if (valid_device == 0) {
        std::cout << "Failed to program any device found, exit!\n";
        exit(EXIT_FAILURE);
    }

    // Allocate Memory in Host Memory
    std::vector<int, aligned_allocator<int>> dev_A((16) * (16));
    std::vector<int, aligned_allocator<int>> dev_B((16) * (16));
    std::vector<int, aligned_allocator<int>> dev_C((16) * (16));

    // Allocate Buffer in Global Memory
    // Buffers are allocated using CL_MEM_USE_HOST_PTR for efficient memory and
    // Device-to-host communication
    OCL_CHECK(err,
              cl::Buffer buffer_A(context,
                                  CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                  (16) * (16) * sizeof(int),
                                  dev_A.data(),
                                  &err));
    OCL_CHECK(err,
              cl::Buffer buffer_B(context,
                                  CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                  (16) * (16) * sizeof(int),
                                  dev_B.data(),
                                  &err));
    OCL_CHECK(err,
              cl::Buffer buffer_C(context,
                                  CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                  (16) * (16) * sizeof(int),
                                  dev_C.data(),
                                  &err));
    
    OCL_CHECK(err,
              err = q.enqueueMigrateMemObjects({buffer_A}, 0));
    OCL_CHECK(err,
              err = q.enqueueMigrateMemObjects({buffer_B}, 0));
    {
      OCL_CHECK(err, err = krnl.setArg(0, buffer_A));
      OCL_CHECK(err, err = krnl.setArg(1, buffer_B));
      OCL_CHECK(err, err = krnl.setArg(2, buffer_C));
      // Launch the Kernel
      OCL_CHECK(err, err = q.enqueueTask(krnl));

      /* Top Function Generation */
      FILE *f = fopen("top.c", "w");
      top_generate(f);
      fclose(f);
      /* Top Function Generation */
    }
    
    OCL_CHECK(err,
              err = q.enqueueMigrateMemObjects({buffer_C}, CL_MIGRATE_MEM_OBJECT_HOST));
    q.finish();
  }

  for (int i = 0; i < I; i++)
    for (int j = 0; j < J; j++) {
      C_golden[i][j] = 0;
      for (int k = 0; k < K; k++) {
        C_golden[i][j] = C_golden[i][j] + A[i][k] * B[j][k];
      }
    }

  int err = 0;
  for (int i = 0; i < I; i++)
    for (int j = 0; j < J; j++) {
      if (abs(C_golden[i][j] - C[i][j]) > 0.001)
        err++;
    }

  if (err)
    printf("Failed with %d errors!\n", err);
  else
    printf("passed!\n");

  return 0;
}
