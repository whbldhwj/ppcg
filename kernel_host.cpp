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

