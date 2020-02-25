#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <CL/opencl.h>
#include "AOCLUtils/aocl_utils.h"
#include "kernel_kernel.h"
using namespace aocl_utils;

#define AOCX_FIEL "krnl.aocx"

#define HOST
#define ACL_ALIGNMENT 64
#ifdef _WIN32
void *acl_aligned_malloc(size_t size) {
    return _aligned_malloc(size, ACL_ALIGNMENT);
}
void acl_aligned_free(void *ptr) {
    _aligned_free(ptr);
}
#else
void *acl_aligned_malloc(size_t size) {
    void *result = NULL;
    if (posix_memalign(&result, ACL_ALIGNMENT, size) != 0)
        printf("acl_aligned_malloc() failed.\n");
    return result;
}
void acl_aligned_free(void *ptr) {
    free(ptr);
}
#endif

void cleanup_host_side_resources();
void cleanup();

#define CHECK(status) \
if (status != CL_SUCCESS) { \
    fprintf(stderr, "error %d in line %d.\n", status, __LINE__); \
    exit(1); \
}

#define CHECK_NO_EXIT(status) \
if (status != CL_SUCCESS) { \
    fprintf(stderr, "error %d in line %d.\n", status, __LINE__); \
}

#include "kernel_top_gen.h"
#include "kernel.h"

int main(int argc, char **argv) {
  data_t A[I][K], B[K][J], C[I][J], C_golden[I][J]; 

  {
    bool use_emulator = false; // control whether the emulator should be used.
    cl_int status;
    cl_platform_id platform = NULL;
    cl_device_id *devices = NULL;
    int NUM_QUEUES_TO_CREATE = 3;
    int NUM_KERNELS_TO_CREATE = 3;
    cl_kernel kernel[NUM_KERNELS_TO_CREATE];
    cl_command_queue cmdQueue[NUM_QUEUES_TO_CREATE];
    cl_mem buf_A = NULL;
    cl_mem buf_B = NULL;
    cl_mem buf_C = NULL;
    int QID_A = 0;
    int QID_B = 1;
    int QID_C = 2;
    int KID_A_0 = 0;
    int KID_B_0 = 1;
    int KID_C_0 = 2;

    // Parse command line arguments
    Options options(argc, argv);
    if (options.has("emulator")) {
        use_emulator = options.get<bool>("emulator")
    }
    if (!setCwdToExeDir()) {
        return false

    // Get the OpenCL platform
    if (use_emulator) {
        platform = findPlatform("Intel(R) FPGA Emulation Platform for OpenCL(TM)");
    } else {
        platform = findPlatform("Intel(R) FPGA SDK for OpenCL(TM)");
    }
    if (platform == NULL) {
        printf("ERROR: Unable to find Intel(R) FPGA OpenCL platform");
        return -1;
    }

    // Discover and initialize the devices
    cl_uint numDevices = 0;
    char buffer[4096];
    unsigned int buf_uint;
    int device_found = 0;
    status = clGetDeviceIDs(platform,
                            CL_DEVICE_TYPE_ALL,
                            0,
                            NULL,
                            &numDevices);
    if (status == CL_SUCCESS) {
        clGetPlatformInfo(platform,
                          CL_PLATFORM_VENDOR,
                          4096,
                          buffer,
                          NULL);
        if (strstr(buffer, "Intel(R)") != NULL) {
            device_found = 1;
        }
        if (device_found) {
            devices = (cl_device_id*) acl_aligned_malloc(numDevices * sizeof(cl_device_id));
            status = clGetDeviceIDs(platform,
                                    CL_DEVICE_TYPE_ALL,
                                    numDevices,
                                    devices,
                                    NULL);
        }
    }
    if (!device_found) {
        printf("failed to find a OpenCL device\n");
        exit(1);
    }
    for (int i = 0; i < numDevices; i++) {
        clGetDeviceInfo(devices[i],
                        CL_DEVICE_NAME,
                        4096,
                        buffer,
                        NULL);
        fprintf(stdout, "\nDevice Name: %s\n", buffer);

        clGetDeviceInfo(devices[i],
                        CL_DEVICE_VENDOR,
                        4096,
                        buffer,
                        NULL);
        fprintf(stdout, "Device Vendor: %s\n", buffer);

        clGetDeviceInfo(devices[i],
                        CL_DEVICE_MAX_COMPUTE_UNITS,
                        sizeof(buf_uint),
                        &buf_uint,
                        NULL);
        fprintf(stdout, "Device Computing Units: %u\n", buf_uint);

        clGetDeviceInfo(devices[i],
                        CL_DEVICE_GLOBAL_MEM_SIZE,
                        sizeof(unsigned long),
                        &buffer,
                        NULL);
        fprintf(stdout, "Global Memory Size: %lu\n", *((unsigned long*)buffer));

        clGetDeviceInfo(devices[i],
                        CL_DEVICE_MAX_MEM_ALLOC_SIZE,
                        sizeof(unsigned long),
                        &buffer,
                        NULL);
        fprintf(stdout, "Global Memory Allocation Size: %lu\n\n", *((unsigned long*)buffer));

    }

    // Create a context
    context = clCreateContext(NULL,
                              1,
                              devices,
                              NULL,
                              NULL,
                              &status); CHECK(status);

    // Create command queues
    for (int i = 0; i < NUM_QUEUES_TO_CREATE; i++) {
        cmdQueue[i] = clCreateCommandQueue(context
                                           devices[0],
                                           CL_QUEUE_PROFILING_ENABLE,
                                           &status); CHECK(status);
    }

    // Create the program from binaries
    size_t binary_length;
    const unsigned char *binary;
    printf("\nAOCX file: %%s\n\n", AOCX_FILE);
    FILE *fp = fopen(AOCX_FILE, "rb");
    if (fp == NULL) {
        printf("Failed to open the AOCX file (fopen).\n");
        return -1;
    }
    fseek(fp, 0, SEEK_END);
    long ftell_sz = ftell(fp);
    if (ftell_sz < 0) {
        printf("ftell returns a negative value.\n");
        fclose(fp);
        return -1;
    } else {
        binary_length = ftell_sz;
    }
    binary = (unsigned char *)malloc(sizeof(unsigned char) * binary_length);
    rewind(fp);
    size_t fread_sz = fread((void *)binary, binary_length, 1, fp);
    if (fread_sz == 0) {
        printf("Failed to read from the AOCX file (fread).\n");
        fclose(fp);
        free(const_char<unsigned char *>(binary))
        return -1;
    }
    fclose(fp);

    program = clCreateProgramWithBinary(context,
                                        1,
                                        devices,
                                        &binary_length,
                                        (const unsigned char **)&binary,
                                        &status,
                                        NULL); CHECK(status);

    status = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (status != CL_SUCCESS) {
        char log[10000] = {0};
        clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, 10000, log, NULL);
        printf("%%s\n", log);
        CHECK(status);
    }

    // Create the kernel
    for (int i = 0; i < NUM_KERNELS_TO_CREATE; i++) {
        kernel[i] = clCreateKernel(program, NULL, &status);
        CHECK(status);
    }
    // Allocate memory in host memory
    int *dev_A = (int*)acl_aligned_malloc((512) * (512) * sizeof(int));
    int *dev_B = (int*)acl_aligned_malloc((512) * (512) * sizeof(int));
    int *dev_C = (int*)acl_aligned_malloc((512) * (512) * sizeof(int));

    memcpy(dev_A, A, (512) * (512) * sizeof(int));
    memcpy(dev_B, B, (512) * (512) * sizeof(int));
    memcpy(dev_C, C, (512) * (512) * sizeof(int));

    // Create device buffers
    buf_A = clCreateBuffer(context
                         CL_MEM_READ_WRITE,
                         (512) * (512) * sizeof(int),
                         NULL,
                         &status); CHECK(status);

    buf_B = clCreateBuffer(context
                         CL_MEM_READ_WRITE,
                         (512) * (512) * sizeof(int),
                         NULL,
                         &status); CHECK(status);

    buf_C = clCreateBuffer(context
                         CL_MEM_READ_WRITE,
                         (512) * (512) * sizeof(int),
                         NULL,
                         &status); CHECK(status);

    status = clEnqueueWriteBuffer(
        cmdQueue[QID_A],
        buf_A,
        CL_TRUE,
        0,
        (512) * (512) * sizeof(int),
        dev_A,
        0,
        NULL,
        NULL); CHECK(status);

    status = clEnqueueWriteBuffer(
        cmdQueue[QID_B],
        buf_B,
        CL_TRUE,
        0,
        (512) * (512) * sizeof(int),
        dev_B,
        0,
        NULL,
        NULL); CHECK(status);

    {
      status = clSetKernelArg(
          kernel[KID_A_0],
          0,
          sizeof(cl_mem),
          (void*)&buf_A); CHECK(status);
      status = clSetKernelArg(
          kernel[KID_B_0],
          0,
          sizeof(cl_mem),
          (void*)&buf_B); CHECK(status);
      status = clSetKernelArg(
          kernel[KID_C_0],
          0,
          sizeof(cl_mem),
          (void*)&buf_C); CHECK(status);

      size_t globalWorkSize[1];
      size_t localWorkSize[1];
      globalWorkSize[0] = 1;
      localWorkSize[0] = 1;

      // Enqueue kernels
      status = clEnqueueNDRangeKernel(
          cmdQueue[QID_A],
          kernel[KID_A_0],
          1,
          NULL,
          globalWorkSize,
          localWorkSize,
          0,
          NULL,
          NULL); CHECK(statis);
      status = clEnqueueNDRangeKernel(
          cmdQueue[QID_B],
          kernel[KID_B_0],
          1,
          NULL,
          globalWorkSize,
          localWorkSize,
          0,
          NULL,
          NULL); CHECK(statis);
      status = clEnqueueNDRangeKernel(
          cmdQueue[QID_C],
          kernel[KID_C_0],
          1,
          NULL,
          globalWorkSize,
          localWorkSize,
          0,
          NULL,
          NULL); CHECK(statis);

      for (int i = 0; i < NUM_QUEUES_TO_CREATE; i++) {
          status = clFinish(cmdQueue[i]); CHECK(status);
      }

      /* Top Function Generation */
      FILE *f = fopen("top.c", "w");
      top_generate(f);
      fclose(f);
      /* Top Function Generation */
    }
    
    status = clEnqueueReadBuffer(
        cmdQueue[QID_C],
        buf_C,
        CL_TRUE,
        0,
        (512) * (512) * sizeof(int),
        dev_C
        0,
        NULL,
        NULL); CHECK(status);

    // clean up resources
    acl_aligned_free(dev_A);
    acl_aligned_free(dev_B);
    acl_aligned_free(dev_C);

    for (int i = 0; i < NUM_KERNELS_TO_CREATE; i++) {
        clReleaseKernel(kernel[i]);
    }

    for (int i = 0; i < NUM_QUEUES_TO_CREATE; i++) {
        clReleaseCommandQueue(cmdQueue[i]);
    }

    clReleaseProgram(program);
    clReleaseContext(context);
    acl_aligned_free(devices);
  }

  for (int i = 0; i < I; i++)
    for (int j = 0; j < J; j++) {
      C_golden[i][j] = 0;
      for (int k = 0; k < K; k++) {
        C_golden[i][j] = C_golden[i][j] + A[i][k] * B[k][j];
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
    prnitf("passed!\n");

  return 0;
}
