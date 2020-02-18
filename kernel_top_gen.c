#include <isl/printer.h>
#include "kernel_top_gen.h"
void top_generate(FILE *f)
{
    isl_ctx *ctx = isl_ctx_alloc();
    isl_printer *p = isl_printer_to_file(ctx, f);
    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, "void kernel0(int *A, int *B, int *C)");
    p = isl_printer_end_line(p);
    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, "{");
    p = isl_printer_end_line(p);
    p = isl_printer_indent(p, 4);
    // array
    // io_L2
    for (int c0 = 0; c0 <= 1; c0 += 1) {
      // io_L1
      for (int c1 = 0; c1 <= 1; c1 += 1) {
        {
          // Print FIFO declarations of module: PE
          p = isl_printer_start_line(p);
          p = isl_printer_print_str(p, "/* PE FIFO */ hls::stream<int> fifo_A_PE");
          p = isl_printer_print_str(p, "_");
          p = isl_printer_print_int(p, c0);
          p = isl_printer_print_str(p, "_");
          p = isl_printer_print_int(p, c1);
          p = isl_printer_print_str(p, ";");
          p = isl_printer_end_line(p);
        }
        if (c1 == 1)
          {
            // Print FIFO declarations of module: PE
            p = isl_printer_start_line(p);
            p = isl_printer_print_str(p, "/* PE FIFO */ hls::stream<int> fifo_A_PE");
            p = isl_printer_print_str(p, "_");
            p = isl_printer_print_int(p, c0);
            p = isl_printer_print_str(p, "_");
            p = isl_printer_print_int(p, c1 + 1);
            p = isl_printer_print_str(p, ";");
            p = isl_printer_end_line(p);
          }
      }
    }

    // array
    // io_L2
    for (int c0 = 0; c0 <= 1; c0 += 1) {
      // io_L1
      for (int c1 = 0; c1 <= 1; c1 += 1) {
        {
          // Print FIFO declarations of module: PE
          p = isl_printer_start_line(p);
          p = isl_printer_print_str(p, "/* PE FIFO */ hls::stream<int> fifo_B_PE");
          p = isl_printer_print_str(p, "_");
          p = isl_printer_print_int(p, c1);
          p = isl_printer_print_str(p, "_");
          p = isl_printer_print_int(p, c0);
          p = isl_printer_print_str(p, ";");
          p = isl_printer_end_line(p);
        }
        if (c1 == 1)
          {
            // Print FIFO declarations of module: PE
            p = isl_printer_start_line(p);
            p = isl_printer_print_str(p, "/* PE FIFO */ hls::stream<int> fifo_B_PE");
            p = isl_printer_print_str(p, "_");
            p = isl_printer_print_int(p, c1 + 1);
            p = isl_printer_print_str(p, "_");
            p = isl_printer_print_int(p, c0);
            p = isl_printer_print_str(p, ";");
            p = isl_printer_end_line(p);
          }
      }
    }

    // array
    // io_L2
    for (int c0 = 0; c0 <= 1; c0 += 1) {
      // io_L1
      for (int c1 = 0; c1 <= 1; c1 += 1)
        {
          // Print FIFO declarations of module: PE
          p = isl_printer_start_line(p);
          p = isl_printer_print_str(p, "/* PE FIFO */ hls::stream<int> fifo_C_drain_PE");
          p = isl_printer_print_str(p, "_");
          p = isl_printer_print_int(p, c1);
          p = isl_printer_print_str(p, "_");
          p = isl_printer_print_int(p, c0);
          p = isl_printer_print_str(p, ";");
          p = isl_printer_end_line(p);
        }
    }

    // array
    // io_L2
    for (int c0 = 0; c0 <= 1; c0 += 1) {
      {
        // Print FIFO declarations of module: A_L2_in_IO
        p = isl_printer_start_line(p);
        p = isl_printer_print_str(p, "/* A_L2_in_IO FIFO */ hls::stream<int> fifo_A_A_L2_in_IO");
        p = isl_printer_print_str(p, "_");
        p = isl_printer_print_int(p, c0);
        p = isl_printer_print_str(p, ";");
        p = isl_printer_end_line(p);
      }
      if (c0 == 1)
        {
          // Print FIFO declarations of module: A_L2_in_IO
          p = isl_printer_start_line(p);
          p = isl_printer_print_str(p, "/* A_L2_in_IO FIFO */ hls::stream<int> fifo_A_A_L2_in_IO");
          p = isl_printer_print_str(p, "_");
          p = isl_printer_print_int(p, c0 + 1);
          p = isl_printer_print_str(p, ";");
          p = isl_printer_end_line(p);
        }
    }

    // array
    // io_L2
    for (int c0 = 0; c0 <= 1; c0 += 1) {
      {
        // Print FIFO declarations of module: B_L2_in_IO
        p = isl_printer_start_line(p);
        p = isl_printer_print_str(p, "/* B_L2_in_IO FIFO */ hls::stream<int> fifo_B_B_L2_in_IO");
        p = isl_printer_print_str(p, "_");
        p = isl_printer_print_int(p, c0);
        p = isl_printer_print_str(p, ";");
        p = isl_printer_end_line(p);
      }
      if (c0 == 1)
        {
          // Print FIFO declarations of module: B_L2_in_IO
          p = isl_printer_start_line(p);
          p = isl_printer_print_str(p, "/* B_L2_in_IO FIFO */ hls::stream<int> fifo_B_B_L2_in_IO");
          p = isl_printer_print_str(p, "_");
          p = isl_printer_print_int(p, c0 + 1);
          p = isl_printer_print_str(p, ";");
          p = isl_printer_end_line(p);
        }
    }

    // array
    // io_L2
    for (int c0 = 0; c0 <= 1; c0 += 1) {
      // io_L1
      for (int c1 = 0; c1 <= 1; c1 += 1) {
        {
          // Print FIFO declarations of module: C_drain_L1_out_IO
          p = isl_printer_start_line(p);
          p = isl_printer_print_str(p, "/* C_drain_L1_out_IO FIFO */ hls::stream<int> fifo_C_drain_C_drain_L1_out_IO");
          p = isl_printer_print_str(p, "_");
          p = isl_printer_print_int(p, c0);
          p = isl_printer_print_str(p, "_");
          p = isl_printer_print_int(p, c1);
          p = isl_printer_print_str(p, ";");
          p = isl_printer_end_line(p);
        }
        if (c1 == 1)
          {
            // Print FIFO declarations of module: C_drain_L1_out_IO
            p = isl_printer_start_line(p);
            p = isl_printer_print_str(p, "/* C_drain_L1_out_IO FIFO */ hls::stream<int> fifo_C_drain_C_drain_L1_out_IO");
            p = isl_printer_print_str(p, "_");
            p = isl_printer_print_int(p, c0);
            p = isl_printer_print_str(p, "_");
            p = isl_printer_print_int(p, c1 + 1);
            p = isl_printer_print_str(p, ";");
            p = isl_printer_end_line(p);
          }
      }
    }

    // array
    // io_L2
    for (int c0 = 0; c0 <= 1; c0 += 1) {
      {
        // Print FIFO declarations of module: C_drain_L2_out_IO
        p = isl_printer_start_line(p);
        p = isl_printer_print_str(p, "/* C_drain_L2_out_IO FIFO */ hls::stream<int> fifo_C_drain_C_drain_L2_out_IO");
        p = isl_printer_print_str(p, "_");
        p = isl_printer_print_int(p, c0);
        p = isl_printer_print_str(p, ";");
        p = isl_printer_end_line(p);
      }
      if (c0 == 1)
        {
          // Print FIFO declarations of module: C_drain_L2_out_IO
          p = isl_printer_start_line(p);
          p = isl_printer_print_str(p, "/* C_drain_L2_out_IO FIFO */ hls::stream<int> fifo_C_drain_C_drain_L2_out_IO");
          p = isl_printer_print_str(p, "_");
          p = isl_printer_print_int(p, c0 + 1);
          p = isl_printer_print_str(p, ";");
          p = isl_printer_end_line(p);
        }
    }

    // array
    for (int c0 = 0; c0 <= 1; c0 += 1)
      for (int c1 = 0; c1 <= 1; c1 += 1)
        {
          // Print calls of module: PE
          p = isl_printer_start_line(p);
          p = isl_printer_print_str(p, "PE(");
          p = isl_printer_end_line(p);
          p = isl_printer_indent(p, 4);
          p = isl_printer_start_line(p);
          p = isl_printer_print_int(p, c0);
          p = isl_printer_print_str(p, ",");
          p = isl_printer_end_line(p);
          p = isl_printer_start_line(p);
          p = isl_printer_print_int(p, c1);
          p = isl_printer_print_str(p, ",");
          p = isl_printer_end_line(p);
          p = isl_printer_start_line(p);
          p = isl_printer_print_str(p, "fifo_A_PE");
          p = isl_printer_print_str(p, "_");
          p = isl_printer_print_int(p, c0);
          p = isl_printer_print_str(p, "_");
          p = isl_printer_print_int(p, c1);
          p = isl_printer_print_str(p, ",");
          p = isl_printer_end_line(p);
          p = isl_printer_start_line(p);
          p = isl_printer_print_str(p, "fifo_A_PE");
          p = isl_printer_print_str(p, "_");
          p = isl_printer_print_int(p, c0);
          p = isl_printer_print_str(p, "_");
          p = isl_printer_print_int(p, c1 + 1);
          p = isl_printer_print_str(p, ",");
          p = isl_printer_end_line(p);
          p = isl_printer_start_line(p);
          p = isl_printer_print_str(p, "fifo_B_PE");
          p = isl_printer_print_str(p, "_");
          p = isl_printer_print_int(p, c0);
          p = isl_printer_print_str(p, "_");
          p = isl_printer_print_int(p, c1);
          p = isl_printer_print_str(p, ",");
          p = isl_printer_end_line(p);
          p = isl_printer_start_line(p);
          p = isl_printer_print_str(p, "fifo_B_PE");
          p = isl_printer_print_str(p, "_");
          p = isl_printer_print_int(p, c0 + 1);
          p = isl_printer_print_str(p, "_");
          p = isl_printer_print_int(p, c1);
          p = isl_printer_print_str(p, ",");
          p = isl_printer_end_line(p);
          p = isl_printer_start_line(p);
          p = isl_printer_print_str(p, "fifo_C_drain_PE");
          p = isl_printer_print_str(p, "_");
          p = isl_printer_print_int(p, c0);
          p = isl_printer_print_str(p, "_");
          p = isl_printer_print_int(p, c1);
          p = isl_printer_end_line(p);
          p = isl_printer_indent(p, -4);
          p = isl_printer_start_line(p);
          p = isl_printer_print_str(p, ");");
          p = isl_printer_end_line(p);
          p = isl_printer_end_line(p);
        }

    // array
    // io_L2
    for (int c0 = 0; c0 <= 1; c0 += 1) {
      {
        // Print calls of module: A_L2_in_IO
        p = isl_printer_start_line(p);
        p = isl_printer_print_str(p, "A_L2_in_IO(");
        p = isl_printer_end_line(p);
        p = isl_printer_indent(p, 4);
        p = isl_printer_start_line(p);
        p = isl_printer_print_int(p, c0);
        p = isl_printer_print_str(p, ",");
        p = isl_printer_end_line(p);
        p = isl_printer_start_line(p);
        p = isl_printer_print_str(p, "fifo_A_A_L2_in_IO");
        p = isl_printer_print_str(p, "_");
        p = isl_printer_print_int(p, c0);
        p = isl_printer_print_str(p, ",");
        p = isl_printer_end_line(p);
        p = isl_printer_start_line(p);
        p = isl_printer_print_str(p, "fifo_A_A_L2_in_IO");
        p = isl_printer_print_str(p, "_");
        p = isl_printer_print_int(p, c0 + 1);
      }
      // io_L1
      for (int c1 = 0; c1 <= 1; c1 += 1)
        if (c1 == 0)
          {
            p = isl_printer_print_str(p, ",");
            p = isl_printer_end_line(p);
            p = isl_printer_start_line(p);
            p = isl_printer_print_str(p, "fifo_A_PE");
            p = isl_printer_print_str(p, "_");
            p = isl_printer_print_int(p, c0);
            p = isl_printer_print_str(p, "_");
            p = isl_printer_print_int(p, c1);
            p = isl_printer_end_line(p);
            p = isl_printer_indent(p, -4);
            p = isl_printer_start_line(p);
            p = isl_printer_print_str(p, ");");
            p = isl_printer_end_line(p);
            p = isl_printer_end_line(p);
          }
    }

    // array
    {
      {
        // Print calls of module: A_L3_in_IO
        p = isl_printer_start_line(p);
        p = isl_printer_print_str(p, "A_L3_in_IO(");
        p = isl_printer_end_line(p);
        p = isl_printer_indent(p, 4);
        p = isl_printer_start_line(p);
        p = isl_printer_print_str(p, "A");
      }
      // io_L2
      for (int c0 = 0; c0 <= 1; c0 += 1)
        if (c0 == 0)
          {
            p = isl_printer_print_str(p, ",");
            p = isl_printer_end_line(p);
            p = isl_printer_start_line(p);
            p = isl_printer_print_str(p, "fifo_A_A_L2_in_IO");
            p = isl_printer_print_str(p, "_");
            p = isl_printer_print_int(p, c0);
            p = isl_printer_end_line(p);
            p = isl_printer_indent(p, -4);
            p = isl_printer_start_line(p);
            p = isl_printer_print_str(p, ");");
            p = isl_printer_end_line(p);
            p = isl_printer_end_line(p);
          }
    }

    // array
    // io_L2
    for (int c0 = 0; c0 <= 1; c0 += 1) {
      {
        // Print calls of module: B_L2_in_IO
        p = isl_printer_start_line(p);
        p = isl_printer_print_str(p, "B_L2_in_IO(");
        p = isl_printer_end_line(p);
        p = isl_printer_indent(p, 4);
        p = isl_printer_start_line(p);
        p = isl_printer_print_int(p, c0);
        p = isl_printer_print_str(p, ",");
        p = isl_printer_end_line(p);
        p = isl_printer_start_line(p);
        p = isl_printer_print_str(p, "fifo_B_B_L2_in_IO");
        p = isl_printer_print_str(p, "_");
        p = isl_printer_print_int(p, c0);
        p = isl_printer_print_str(p, ",");
        p = isl_printer_end_line(p);
        p = isl_printer_start_line(p);
        p = isl_printer_print_str(p, "fifo_B_B_L2_in_IO");
        p = isl_printer_print_str(p, "_");
        p = isl_printer_print_int(p, c0 + 1);
      }
      // io_L1
      for (int c1 = 0; c1 <= 1; c1 += 1)
        if (c1 == 0)
          {
            p = isl_printer_print_str(p, ",");
            p = isl_printer_end_line(p);
            p = isl_printer_start_line(p);
            p = isl_printer_print_str(p, "fifo_B_PE");
            p = isl_printer_print_str(p, "_");
            p = isl_printer_print_int(p, c1);
            p = isl_printer_print_str(p, "_");
            p = isl_printer_print_int(p, c0);
            p = isl_printer_end_line(p);
            p = isl_printer_indent(p, -4);
            p = isl_printer_start_line(p);
            p = isl_printer_print_str(p, ");");
            p = isl_printer_end_line(p);
            p = isl_printer_end_line(p);
          }
    }

    // array
    {
      {
        // Print calls of module: B_L3_in_IO
        p = isl_printer_start_line(p);
        p = isl_printer_print_str(p, "B_L3_in_IO(");
        p = isl_printer_end_line(p);
        p = isl_printer_indent(p, 4);
        p = isl_printer_start_line(p);
        p = isl_printer_print_str(p, "B");
      }
      // io_L2
      for (int c0 = 0; c0 <= 1; c0 += 1)
        if (c0 == 0)
          {
            p = isl_printer_print_str(p, ",");
            p = isl_printer_end_line(p);
            p = isl_printer_start_line(p);
            p = isl_printer_print_str(p, "fifo_B_B_L2_in_IO");
            p = isl_printer_print_str(p, "_");
            p = isl_printer_print_int(p, c0);
            p = isl_printer_end_line(p);
            p = isl_printer_indent(p, -4);
            p = isl_printer_start_line(p);
            p = isl_printer_print_str(p, ");");
            p = isl_printer_end_line(p);
            p = isl_printer_end_line(p);
          }
    }

    // array
    // io_L2
    for (int c0 = 0; c0 <= 1; c0 += 1) {
      // io_L1
      for (int c1 = 0; c1 <= 1; c1 += 1) {
        {
          // Print calls of module: C_drain_L1_out_IO
          p = isl_printer_start_line(p);
          p = isl_printer_print_str(p, "C_drain_L1_out_IO(");
          p = isl_printer_end_line(p);
          p = isl_printer_indent(p, 4);
          p = isl_printer_start_line(p);
          p = isl_printer_print_int(p, c0);
          p = isl_printer_print_str(p, ",");
          p = isl_printer_end_line(p);
          p = isl_printer_start_line(p);
          p = isl_printer_print_int(p, c1);
          p = isl_printer_print_str(p, ",");
          p = isl_printer_end_line(p);
          p = isl_printer_start_line(p);
          p = isl_printer_print_str(p, "fifo_C_drain_C_drain_L1_out_IO");
          p = isl_printer_print_str(p, "_");
          p = isl_printer_print_int(p, c0);
          p = isl_printer_print_str(p, "_");
          p = isl_printer_print_int(p, c1);
          p = isl_printer_print_str(p, ",");
          p = isl_printer_end_line(p);
          p = isl_printer_start_line(p);
          p = isl_printer_print_str(p, "fifo_C_drain_C_drain_L1_out_IO");
          p = isl_printer_print_str(p, "_");
          p = isl_printer_print_int(p, c0);
          p = isl_printer_print_str(p, "_");
          p = isl_printer_print_int(p, c1 + 1);
        }
        {
          p = isl_printer_print_str(p, ",");
          p = isl_printer_end_line(p);
          p = isl_printer_start_line(p);
          p = isl_printer_print_str(p, "fifo_C_drain_PE");
          p = isl_printer_print_str(p, "_");
          p = isl_printer_print_int(p, c1);
          p = isl_printer_print_str(p, "_");
          p = isl_printer_print_int(p, c0);
          p = isl_printer_end_line(p);
          p = isl_printer_indent(p, -4);
          p = isl_printer_start_line(p);
          p = isl_printer_print_str(p, ");");
          p = isl_printer_end_line(p);
          p = isl_printer_end_line(p);
        }
      }
    }

    // array
    // io_L2
    for (int c0 = 0; c0 <= 1; c0 += 1) {
      {
        // Print calls of module: C_drain_L2_out_IO
        p = isl_printer_start_line(p);
        p = isl_printer_print_str(p, "C_drain_L2_out_IO(");
        p = isl_printer_end_line(p);
        p = isl_printer_indent(p, 4);
        p = isl_printer_start_line(p);
        p = isl_printer_print_int(p, c0);
        p = isl_printer_print_str(p, ",");
        p = isl_printer_end_line(p);
        p = isl_printer_start_line(p);
        p = isl_printer_print_str(p, "fifo_C_drain_C_drain_L2_out_IO");
        p = isl_printer_print_str(p, "_");
        p = isl_printer_print_int(p, c0);
        p = isl_printer_print_str(p, ",");
        p = isl_printer_end_line(p);
        p = isl_printer_start_line(p);
        p = isl_printer_print_str(p, "fifo_C_drain_C_drain_L2_out_IO");
        p = isl_printer_print_str(p, "_");
        p = isl_printer_print_int(p, c0 + 1);
      }
      // io_L1
      for (int c1 = 0; c1 <= 1; c1 += 1)
        if (c1 == 1)
          {
            p = isl_printer_print_str(p, ",");
            p = isl_printer_end_line(p);
            p = isl_printer_start_line(p);
            p = isl_printer_print_str(p, "fifo_C_drain_C_drain_L1_out_IO");
            p = isl_printer_print_str(p, "_");
            p = isl_printer_print_int(p, c0);
            p = isl_printer_print_str(p, "_");
            p = isl_printer_print_int(p, c1 + 1);
            p = isl_printer_end_line(p);
            p = isl_printer_indent(p, -4);
            p = isl_printer_start_line(p);
            p = isl_printer_print_str(p, ");");
            p = isl_printer_end_line(p);
            p = isl_printer_end_line(p);
          }
    }

    // array
    {
      {
        // Print calls of module: C_drain_L3_out_IO
        p = isl_printer_start_line(p);
        p = isl_printer_print_str(p, "C_drain_L3_out_IO(");
        p = isl_printer_end_line(p);
        p = isl_printer_indent(p, 4);
        p = isl_printer_start_line(p);
        p = isl_printer_print_str(p, "C");
      }
      // io_L2
      for (int c0 = 0; c0 <= 1; c0 += 1)
        if (c0 == 1)
          {
            p = isl_printer_print_str(p, ",");
            p = isl_printer_end_line(p);
            p = isl_printer_start_line(p);
            p = isl_printer_print_str(p, "fifo_C_drain_C_drain_L2_out_IO");
            p = isl_printer_print_str(p, "_");
            p = isl_printer_print_int(p, c0 + 1);
            p = isl_printer_end_line(p);
            p = isl_printer_indent(p, -4);
            p = isl_printer_start_line(p);
            p = isl_printer_print_str(p, ");");
            p = isl_printer_end_line(p);
            p = isl_printer_end_line(p);
          }
    }

    p = isl_printer_indent(p, -4);
    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, "}");
    p = isl_printer_end_line(p);
    isl_printer_free(p);
    isl_ctx_free(ctx);
}

int main()
{
  FILE *f = fopen("temp.c", "w");
  top_generate(f);
}
