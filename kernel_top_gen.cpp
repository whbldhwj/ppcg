#include <isl/printer.h>
#include "kernel_top_gen.h"
void top_generate(FILE *f)
{
    isl_ctx *ctx = isl_ctx_alloc();
    isl_printer *p = isl_printer_to_file(ctx, f);
    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, "extern \"C\" {");
    p = isl_printer_end_line(p);
    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, "void kernel0(A_t4 *A, B_t4 *B, C_t2 *C)");
    p = isl_printer_end_line(p);
    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, "{");
    p = isl_printer_end_line(p);
    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, "#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem");
    p = isl_printer_end_line(p);
    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, "#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem");
    p = isl_printer_end_line(p);
    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, "#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem");
    p = isl_printer_end_line(p);
    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, "#pragma HLS INTERFACE s_axilite port=A bundle=control");
    p = isl_printer_end_line(p);
    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, "#pragma HLS INTERFACE s_axilite port=B bundle=control");
    p = isl_printer_end_line(p);
    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, "#pragma HLS INTERFACE s_axilite port=C bundle=control");
    p = isl_printer_end_line(p);
    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, "#pragma HLS INTERFACE s_axilite port=return bundle=control");
    p = isl_printer_end_line(p);
    p = isl_printer_end_line(p);
    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, "#pragma HLS DATAFLOW");
    p = isl_printer_end_line(p);
    p = isl_printer_end_line(p);
    p = isl_printer_indent(p, 4);
    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, "/* FIFO Declaration */");
    p = isl_printer_end_line(p);
    // array
    // io_L3
    for (int c0 = 0; c0 <= 1; c0 += 1) {
      {
        // Print channel declarations of module: A_IO_L2_in
        p = isl_printer_start_line(p);
        p = isl_printer_print_str(p, "/* A_IO_L2_in fifo */ hls::stream<A_t4> fifo_A_A_IO_L2_in");
        p = isl_printer_print_str(p, "_");
        p = isl_printer_print_int(p, c0);
        p = isl_printer_print_str(p, ";");
        p = isl_printer_end_line(p);
        p = isl_printer_start_line(p);
        p = isl_printer_print_str(p, "#pragma HLS STREAM variable=fifo_A_A_IO_L2_in");
        p = isl_printer_print_str(p, "_");
        p = isl_printer_print_int(p, c0);
        p = isl_printer_print_str(p, " depth=1");
        p = isl_printer_end_line(p);
      }
      if (c0 == 1)
        {
          // Print channel declarations of module: A_IO_L2_in
          p = isl_printer_start_line(p);
          p = isl_printer_print_str(p, "/* A_IO_L2_in fifo */ hls::stream<A_t4> fifo_A_A_IO_L2_in");
          p = isl_printer_print_str(p, "_");
          p = isl_printer_print_int(p, c0 + 1);
          p = isl_printer_print_str(p, ";");
          p = isl_printer_end_line(p);
          p = isl_printer_start_line(p);
          p = isl_printer_print_str(p, "#pragma HLS STREAM variable=fifo_A_A_IO_L2_in");
          p = isl_printer_print_str(p, "_");
          p = isl_printer_print_int(p, c0 + 1);
          p = isl_printer_print_str(p, " depth=1");
          p = isl_printer_end_line(p);
        }
    }

    // array
    // io_L3
    for (int c0 = 0; c0 <= 1; c0 += 1) {
      {
        // Print channel declarations of module: B_IO_L2_in
        p = isl_printer_start_line(p);
        p = isl_printer_print_str(p, "/* B_IO_L2_in fifo */ hls::stream<B_t4> fifo_B_B_IO_L2_in");
        p = isl_printer_print_str(p, "_");
        p = isl_printer_print_int(p, c0);
        p = isl_printer_print_str(p, ";");
        p = isl_printer_end_line(p);
        p = isl_printer_start_line(p);
        p = isl_printer_print_str(p, "#pragma HLS STREAM variable=fifo_B_B_IO_L2_in");
        p = isl_printer_print_str(p, "_");
        p = isl_printer_print_int(p, c0);
        p = isl_printer_print_str(p, " depth=1");
        p = isl_printer_end_line(p);
      }
      if (c0 == 1)
        {
          // Print channel declarations of module: B_IO_L2_in
          p = isl_printer_start_line(p);
          p = isl_printer_print_str(p, "/* B_IO_L2_in fifo */ hls::stream<B_t4> fifo_B_B_IO_L2_in");
          p = isl_printer_print_str(p, "_");
          p = isl_printer_print_int(p, c0 + 1);
          p = isl_printer_print_str(p, ";");
          p = isl_printer_end_line(p);
          p = isl_printer_start_line(p);
          p = isl_printer_print_str(p, "#pragma HLS STREAM variable=fifo_B_B_IO_L2_in");
          p = isl_printer_print_str(p, "_");
          p = isl_printer_print_int(p, c0 + 1);
          p = isl_printer_print_str(p, " depth=1");
          p = isl_printer_end_line(p);
        }
    }

    // array
    for (int c0 = 0; c0 <= 1; c0 += 1)
      for (int c1 = 0; c1 <= 1; c1 += 1) {
        // io_L1
        {
          {
            // Print channel declarations of module: PE
            p = isl_printer_start_line(p);
            p = isl_printer_print_str(p, "/* PE fifo */ hls::stream<A_t2> fifo_A_PE");
            p = isl_printer_print_str(p, "_");
            p = isl_printer_print_int(p, c0);
            p = isl_printer_print_str(p, "_");
            p = isl_printer_print_int(p, c1);
            p = isl_printer_print_str(p, ";");
            p = isl_printer_end_line(p);
            p = isl_printer_start_line(p);
            p = isl_printer_print_str(p, "#pragma HLS STREAM variable=fifo_A_PE");
            p = isl_printer_print_str(p, "_");
            p = isl_printer_print_int(p, c0);
            p = isl_printer_print_str(p, "_");
            p = isl_printer_print_int(p, c1);
            p = isl_printer_print_str(p, " depth=1");
            p = isl_printer_end_line(p);
          }
          if (c1 == 1)
            {
              // Print channel declarations of module: PE
              p = isl_printer_start_line(p);
              p = isl_printer_print_str(p, "/* PE fifo */ hls::stream<A_t2> fifo_A_PE");
              p = isl_printer_print_str(p, "_");
              p = isl_printer_print_int(p, c0);
              p = isl_printer_print_str(p, "_");
              p = isl_printer_print_int(p, c1 + 1);
              p = isl_printer_print_str(p, ";");
              p = isl_printer_end_line(p);
              p = isl_printer_start_line(p);
              p = isl_printer_print_str(p, "#pragma HLS STREAM variable=fifo_A_PE");
              p = isl_printer_print_str(p, "_");
              p = isl_printer_print_int(p, c0);
              p = isl_printer_print_str(p, "_");
              p = isl_printer_print_int(p, c1 + 1);
              p = isl_printer_print_str(p, " depth=1");
              p = isl_printer_end_line(p);
            }
        }
      }

    // array
    for (int c0 = 0; c0 <= 1; c0 += 1)
      for (int c1 = 0; c1 <= 1; c1 += 1) {
        // io_L1
        {
          {
            // Print channel declarations of module: PE
            p = isl_printer_start_line(p);
            p = isl_printer_print_str(p, "/* PE fifo */ hls::stream<B_t2> fifo_B_PE");
            p = isl_printer_print_str(p, "_");
            p = isl_printer_print_int(p, c1);
            p = isl_printer_print_str(p, "_");
            p = isl_printer_print_int(p, c0);
            p = isl_printer_print_str(p, ";");
            p = isl_printer_end_line(p);
            p = isl_printer_start_line(p);
            p = isl_printer_print_str(p, "#pragma HLS STREAM variable=fifo_B_PE");
            p = isl_printer_print_str(p, "_");
            p = isl_printer_print_int(p, c1);
            p = isl_printer_print_str(p, "_");
            p = isl_printer_print_int(p, c0);
            p = isl_printer_print_str(p, " depth=1");
            p = isl_printer_end_line(p);
          }
          if (c1 == 1)
            {
              // Print channel declarations of module: PE
              p = isl_printer_start_line(p);
              p = isl_printer_print_str(p, "/* PE fifo */ hls::stream<B_t2> fifo_B_PE");
              p = isl_printer_print_str(p, "_");
              p = isl_printer_print_int(p, c1 + 1);
              p = isl_printer_print_str(p, "_");
              p = isl_printer_print_int(p, c0);
              p = isl_printer_print_str(p, ";");
              p = isl_printer_end_line(p);
              p = isl_printer_start_line(p);
              p = isl_printer_print_str(p, "#pragma HLS STREAM variable=fifo_B_PE");
              p = isl_printer_print_str(p, "_");
              p = isl_printer_print_int(p, c1 + 1);
              p = isl_printer_print_str(p, "_");
              p = isl_printer_print_int(p, c0);
              p = isl_printer_print_str(p, " depth=1");
              p = isl_printer_end_line(p);
            }
        }
      }

    // array
    for (int c0 = 0; c0 <= 1; c0 += 1)
      for (int c1 = 0; c1 <= 1; c1 += 1) {
        // io_L1
        {
          // Print channel declarations of module: PE
          p = isl_printer_start_line(p);
          p = isl_printer_print_str(p, "/* PE fifo */ hls::stream<int> fifo_C_drain_PE");
          p = isl_printer_print_str(p, "_");
          p = isl_printer_print_int(p, c1);
          p = isl_printer_print_str(p, "_");
          p = isl_printer_print_int(p, c0);
          p = isl_printer_print_str(p, ";");
          p = isl_printer_end_line(p);
          p = isl_printer_start_line(p);
          p = isl_printer_print_str(p, "#pragma HLS STREAM variable=fifo_C_drain_PE");
          p = isl_printer_print_str(p, "_");
          p = isl_printer_print_int(p, c1);
          p = isl_printer_print_str(p, "_");
          p = isl_printer_print_int(p, c0);
          p = isl_printer_print_str(p, " depth=1");
          p = isl_printer_end_line(p);
        }
      }

    // array
    // io_L3
    for (int c0 = 0; c0 <= 1; c0 += 1) {
      // io_L2
      for (int c1 = 0; c1 <= 1; c1 += 1) {
        {
          // Print channel declarations of module: C_drain_IO_L1_out
          p = isl_printer_start_line(p);
          p = isl_printer_print_str(p, "/* C_drain_IO_L1_out fifo */ hls::stream<C_t2> fifo_C_drain_C_drain_IO_L1_out");
          p = isl_printer_print_str(p, "_");
          p = isl_printer_print_int(p, c0);
          p = isl_printer_print_str(p, "_");
          p = isl_printer_print_int(p, c1);
          p = isl_printer_print_str(p, ";");
          p = isl_printer_end_line(p);
          p = isl_printer_start_line(p);
          p = isl_printer_print_str(p, "#pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L1_out");
          p = isl_printer_print_str(p, "_");
          p = isl_printer_print_int(p, c0);
          p = isl_printer_print_str(p, "_");
          p = isl_printer_print_int(p, c1);
          p = isl_printer_print_str(p, " depth=1");
          p = isl_printer_end_line(p);
        }
        if (c1 == 1)
          {
            // Print channel declarations of module: C_drain_IO_L1_out
            p = isl_printer_start_line(p);
            p = isl_printer_print_str(p, "/* C_drain_IO_L1_out fifo */ hls::stream<C_t2> fifo_C_drain_C_drain_IO_L1_out");
            p = isl_printer_print_str(p, "_");
            p = isl_printer_print_int(p, c0);
            p = isl_printer_print_str(p, "_");
            p = isl_printer_print_int(p, c1 + 1);
            p = isl_printer_print_str(p, ";");
            p = isl_printer_end_line(p);
            p = isl_printer_start_line(p);
            p = isl_printer_print_str(p, "#pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L1_out");
            p = isl_printer_print_str(p, "_");
            p = isl_printer_print_int(p, c0);
            p = isl_printer_print_str(p, "_");
            p = isl_printer_print_int(p, c1 + 1);
            p = isl_printer_print_str(p, " depth=1");
            p = isl_printer_end_line(p);
          }
      }
    }

    // array
    // io_L3
    for (int c0 = 0; c0 <= 1; c0 += 1) {
      {
        // Print channel declarations of module: C_drain_IO_L2_out
        p = isl_printer_start_line(p);
        p = isl_printer_print_str(p, "/* C_drain_IO_L2_out fifo */ hls::stream<C_t2> fifo_C_drain_C_drain_IO_L2_out");
        p = isl_printer_print_str(p, "_");
        p = isl_printer_print_int(p, c0);
        p = isl_printer_print_str(p, ";");
        p = isl_printer_end_line(p);
        p = isl_printer_start_line(p);
        p = isl_printer_print_str(p, "#pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L2_out");
        p = isl_printer_print_str(p, "_");
        p = isl_printer_print_int(p, c0);
        p = isl_printer_print_str(p, " depth=1");
        p = isl_printer_end_line(p);
      }
      if (c0 == 1)
        {
          // Print channel declarations of module: C_drain_IO_L2_out
          p = isl_printer_start_line(p);
          p = isl_printer_print_str(p, "/* C_drain_IO_L2_out fifo */ hls::stream<C_t2> fifo_C_drain_C_drain_IO_L2_out");
          p = isl_printer_print_str(p, "_");
          p = isl_printer_print_int(p, c0 + 1);
          p = isl_printer_print_str(p, ";");
          p = isl_printer_end_line(p);
          p = isl_printer_start_line(p);
          p = isl_printer_print_str(p, "#pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L2_out");
          p = isl_printer_print_str(p, "_");
          p = isl_printer_print_int(p, c0 + 1);
          p = isl_printer_print_str(p, " depth=1");
          p = isl_printer_end_line(p);
        }
    }

    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, "/* FIFO Declaration */");
    p = isl_printer_end_line(p);
    p = isl_printer_end_line(p);
    // array
    {
      {
        p = isl_printer_start_line(p);
        p = isl_printer_print_str(p, "/* Module Call */");
        p = isl_printer_end_line(p);
        // Print calls of module: A_IO_L3_in
        p = isl_printer_start_line(p);
        p = isl_printer_print_str(p, "A_IO_L3_in(");
        p = isl_printer_end_line(p);
        p = isl_printer_indent(p, 4);
        p = isl_printer_start_line(p);
        p = isl_printer_print_str(p, "/* array */ A");
      }
      // io_L3
      for (int c0 = 0; c0 <= 1; c0 += 1)
        if (c0 == 0)
          {
            p = isl_printer_print_str(p, ",");
            p = isl_printer_end_line(p);
            p = isl_printer_start_line(p);
            p = isl_printer_print_str(p, "/* fifo */ ");
            p = isl_printer_print_str(p, "fifo_A_A_IO_L2_in");
            p = isl_printer_print_str(p, "_");
            p = isl_printer_print_int(p, c0);
            p = isl_printer_end_line(p);
            p = isl_printer_indent(p, -4);
            p = isl_printer_start_line(p);
            p = isl_printer_print_str(p, ");");
            p = isl_printer_end_line(p);
            p = isl_printer_start_line(p);
            p = isl_printer_print_str(p, "/* Module Call */");
            p = isl_printer_end_line(p);
            p = isl_printer_end_line(p);
          }
    }

    // array
    // io_L3
    for (int c0 = 0; c0 <= 1; c0 += 1) {
      {
        p = isl_printer_start_line(p);
        p = isl_printer_print_str(p, "/* Module Call */");
        p = isl_printer_end_line(p);
        // Print calls of module: A_IO_L2_in
        p = isl_printer_start_line(p);
        p = isl_printer_print_str(p, "A_IO_L2_in(");
        p = isl_printer_end_line(p);
        p = isl_printer_indent(p, 4);
        p = isl_printer_start_line(p);
        p = isl_printer_print_str(p, "/* module id */ ");
        p = isl_printer_print_int(p, c0);
        p = isl_printer_print_str(p, ",");
        p = isl_printer_end_line(p);
        p = isl_printer_start_line(p);
        p = isl_printer_print_str(p, "/* fifo */ ");
        p = isl_printer_print_str(p, "fifo_A_A_IO_L2_in");
        p = isl_printer_print_str(p, "_");
        p = isl_printer_print_int(p, c0);
        p = isl_printer_print_str(p, ",");
        p = isl_printer_end_line(p);
        p = isl_printer_start_line(p);
        p = isl_printer_print_str(p, "/* fifo */ ");
        p = isl_printer_print_str(p, "fifo_A_A_IO_L2_in");
        p = isl_printer_print_str(p, "_");
        p = isl_printer_print_int(p, c0 + 1);
      }
      // io_L2
      for (int c1 = 0; c1 <= 1; c1 += 1)
        if (c1 == 0)
          {
            p = isl_printer_print_str(p, ",");
            p = isl_printer_end_line(p);
            p = isl_printer_start_line(p);
            p = isl_printer_print_str(p, "/* fifo */ ");
            p = isl_printer_print_str(p, "fifo_A_PE");
            p = isl_printer_print_str(p, "_");
            p = isl_printer_print_int(p, c0);
            p = isl_printer_print_str(p, "_");
            p = isl_printer_print_int(p, 0);
            p = isl_printer_end_line(p);
            p = isl_printer_indent(p, -4);
            p = isl_printer_start_line(p);
            p = isl_printer_print_str(p, ");");
            p = isl_printer_end_line(p);
            p = isl_printer_start_line(p);
            p = isl_printer_print_str(p, "/* Module Call */");
            p = isl_printer_end_line(p);
            p = isl_printer_end_line(p);
          }
    }

    // array
    {
      {
        p = isl_printer_start_line(p);
        p = isl_printer_print_str(p, "/* Module Call */");
        p = isl_printer_end_line(p);
        // Print calls of module: B_IO_L3_in
        p = isl_printer_start_line(p);
        p = isl_printer_print_str(p, "B_IO_L3_in(");
        p = isl_printer_end_line(p);
        p = isl_printer_indent(p, 4);
        p = isl_printer_start_line(p);
        p = isl_printer_print_str(p, "/* array */ B");
      }
      // io_L3
      for (int c0 = 0; c0 <= 1; c0 += 1)
        if (c0 == 0)
          {
            p = isl_printer_print_str(p, ",");
            p = isl_printer_end_line(p);
            p = isl_printer_start_line(p);
            p = isl_printer_print_str(p, "/* fifo */ ");
            p = isl_printer_print_str(p, "fifo_B_B_IO_L2_in");
            p = isl_printer_print_str(p, "_");
            p = isl_printer_print_int(p, c0);
            p = isl_printer_end_line(p);
            p = isl_printer_indent(p, -4);
            p = isl_printer_start_line(p);
            p = isl_printer_print_str(p, ");");
            p = isl_printer_end_line(p);
            p = isl_printer_start_line(p);
            p = isl_printer_print_str(p, "/* Module Call */");
            p = isl_printer_end_line(p);
            p = isl_printer_end_line(p);
          }
    }

    // array
    // io_L3
    for (int c0 = 0; c0 <= 1; c0 += 1) {
      {
        p = isl_printer_start_line(p);
        p = isl_printer_print_str(p, "/* Module Call */");
        p = isl_printer_end_line(p);
        // Print calls of module: B_IO_L2_in
        p = isl_printer_start_line(p);
        p = isl_printer_print_str(p, "B_IO_L2_in(");
        p = isl_printer_end_line(p);
        p = isl_printer_indent(p, 4);
        p = isl_printer_start_line(p);
        p = isl_printer_print_str(p, "/* module id */ ");
        p = isl_printer_print_int(p, c0);
        p = isl_printer_print_str(p, ",");
        p = isl_printer_end_line(p);
        p = isl_printer_start_line(p);
        p = isl_printer_print_str(p, "/* fifo */ ");
        p = isl_printer_print_str(p, "fifo_B_B_IO_L2_in");
        p = isl_printer_print_str(p, "_");
        p = isl_printer_print_int(p, c0);
        p = isl_printer_print_str(p, ",");
        p = isl_printer_end_line(p);
        p = isl_printer_start_line(p);
        p = isl_printer_print_str(p, "/* fifo */ ");
        p = isl_printer_print_str(p, "fifo_B_B_IO_L2_in");
        p = isl_printer_print_str(p, "_");
        p = isl_printer_print_int(p, c0 + 1);
      }
      // io_L2
      for (int c1 = 0; c1 <= 1; c1 += 1)
        if (c1 == 0)
          {
            p = isl_printer_print_str(p, ",");
            p = isl_printer_end_line(p);
            p = isl_printer_start_line(p);
            p = isl_printer_print_str(p, "/* fifo */ ");
            p = isl_printer_print_str(p, "fifo_B_PE");
            p = isl_printer_print_str(p, "_");
            p = isl_printer_print_int(p, 0);
            p = isl_printer_print_str(p, "_");
            p = isl_printer_print_int(p, c0);
            p = isl_printer_end_line(p);
            p = isl_printer_indent(p, -4);
            p = isl_printer_start_line(p);
            p = isl_printer_print_str(p, ");");
            p = isl_printer_end_line(p);
            p = isl_printer_start_line(p);
            p = isl_printer_print_str(p, "/* Module Call */");
            p = isl_printer_end_line(p);
            p = isl_printer_end_line(p);
          }
    }

    // array
    for (int c0 = 0; c0 <= 1; c0 += 1)
      for (int c1 = 0; c1 <= 1; c1 += 1)
        {
          p = isl_printer_start_line(p);
          p = isl_printer_print_str(p, "/* Module Call */");
          p = isl_printer_end_line(p);
          // Print calls of module: PE
          p = isl_printer_start_line(p);
          p = isl_printer_print_str(p, "PE(");
          p = isl_printer_end_line(p);
          p = isl_printer_indent(p, 4);
          p = isl_printer_start_line(p);
          p = isl_printer_print_str(p, "/* module id */ ");
          p = isl_printer_print_int(p, c0);
          p = isl_printer_print_str(p, ",");
          p = isl_printer_end_line(p);
          p = isl_printer_start_line(p);
          p = isl_printer_print_str(p, "/* module id */ ");
          p = isl_printer_print_int(p, c1);
          p = isl_printer_print_str(p, ",");
          p = isl_printer_end_line(p);
          p = isl_printer_start_line(p);
          p = isl_printer_print_str(p, "/* fifo */ ");
          p = isl_printer_print_str(p, "fifo_A_PE");
          p = isl_printer_print_str(p, "_");
          p = isl_printer_print_int(p, c0);
          p = isl_printer_print_str(p, "_");
          p = isl_printer_print_int(p, c1);
          p = isl_printer_print_str(p, ",");
          p = isl_printer_end_line(p);
          p = isl_printer_start_line(p);
          p = isl_printer_print_str(p, "/* fifo */ ");
          p = isl_printer_print_str(p, "fifo_A_PE");
          p = isl_printer_print_str(p, "_");
          p = isl_printer_print_int(p, c0);
          p = isl_printer_print_str(p, "_");
          p = isl_printer_print_int(p, c1 + 1);
          p = isl_printer_print_str(p, ",");
          p = isl_printer_end_line(p);
          p = isl_printer_start_line(p);
          p = isl_printer_print_str(p, "/* fifo */ ");
          p = isl_printer_print_str(p, "fifo_B_PE");
          p = isl_printer_print_str(p, "_");
          p = isl_printer_print_int(p, c0);
          p = isl_printer_print_str(p, "_");
          p = isl_printer_print_int(p, c1);
          p = isl_printer_print_str(p, ",");
          p = isl_printer_end_line(p);
          p = isl_printer_start_line(p);
          p = isl_printer_print_str(p, "/* fifo */ ");
          p = isl_printer_print_str(p, "fifo_B_PE");
          p = isl_printer_print_str(p, "_");
          p = isl_printer_print_int(p, c0 + 1);
          p = isl_printer_print_str(p, "_");
          p = isl_printer_print_int(p, c1);
          p = isl_printer_print_str(p, ",");
          p = isl_printer_end_line(p);
          p = isl_printer_start_line(p);
          p = isl_printer_print_str(p, "/* fifo */ ");
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
          p = isl_printer_start_line(p);
          p = isl_printer_print_str(p, "/* Module Call */");
          p = isl_printer_end_line(p);
          p = isl_printer_end_line(p);
        }

    // array
    // io_L3
    for (int c0 = 0; c0 <= 1; c0 += 1) {
      // io_L2
      for (int c1 = 0; c1 <= 1; c1 += 1) {
        {
          p = isl_printer_start_line(p);
          p = isl_printer_print_str(p, "/* Module Call */");
          p = isl_printer_end_line(p);
          // Print calls of module: C_drain_IO_L1_out
          p = isl_printer_start_line(p);
          p = isl_printer_print_str(p, "C_drain_IO_L1_out(");
          p = isl_printer_end_line(p);
          p = isl_printer_indent(p, 4);
          p = isl_printer_start_line(p);
          p = isl_printer_print_str(p, "/* module id */ ");
          p = isl_printer_print_int(p, c0);
          p = isl_printer_print_str(p, ",");
          p = isl_printer_end_line(p);
          p = isl_printer_start_line(p);
          p = isl_printer_print_str(p, "/* module id */ ");
          p = isl_printer_print_int(p, c1);
          p = isl_printer_print_str(p, ",");
          p = isl_printer_end_line(p);
          p = isl_printer_start_line(p);
          p = isl_printer_print_str(p, "/* fifo */ ");
          p = isl_printer_print_str(p, "fifo_C_drain_C_drain_IO_L1_out");
          p = isl_printer_print_str(p, "_");
          p = isl_printer_print_int(p, c0);
          p = isl_printer_print_str(p, "_");
          p = isl_printer_print_int(p, c1);
          p = isl_printer_print_str(p, ",");
          p = isl_printer_end_line(p);
          p = isl_printer_start_line(p);
          p = isl_printer_print_str(p, "/* fifo */ ");
          p = isl_printer_print_str(p, "fifo_C_drain_C_drain_IO_L1_out");
          p = isl_printer_print_str(p, "_");
          p = isl_printer_print_int(p, c0);
          p = isl_printer_print_str(p, "_");
          p = isl_printer_print_int(p, c1 + 1);
        }
        {
          p = isl_printer_print_str(p, ",");
          p = isl_printer_end_line(p);
          p = isl_printer_start_line(p);
          p = isl_printer_print_str(p, "/* fifo */ ");
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
          p = isl_printer_start_line(p);
          p = isl_printer_print_str(p, "/* Module Call */");
          p = isl_printer_end_line(p);
          p = isl_printer_end_line(p);
        }
      }
    }

    // array
    // io_L3
    for (int c0 = 0; c0 <= 1; c0 += 1) {
      {
        p = isl_printer_start_line(p);
        p = isl_printer_print_str(p, "/* Module Call */");
        p = isl_printer_end_line(p);
        // Print calls of module: C_drain_IO_L2_out
        p = isl_printer_start_line(p);
        p = isl_printer_print_str(p, "C_drain_IO_L2_out(");
        p = isl_printer_end_line(p);
        p = isl_printer_indent(p, 4);
        p = isl_printer_start_line(p);
        p = isl_printer_print_str(p, "/* module id */ ");
        p = isl_printer_print_int(p, c0);
        p = isl_printer_print_str(p, ",");
        p = isl_printer_end_line(p);
        p = isl_printer_start_line(p);
        p = isl_printer_print_str(p, "/* fifo */ ");
        p = isl_printer_print_str(p, "fifo_C_drain_C_drain_IO_L2_out");
        p = isl_printer_print_str(p, "_");
        p = isl_printer_print_int(p, c0);
        p = isl_printer_print_str(p, ",");
        p = isl_printer_end_line(p);
        p = isl_printer_start_line(p);
        p = isl_printer_print_str(p, "/* fifo */ ");
        p = isl_printer_print_str(p, "fifo_C_drain_C_drain_IO_L2_out");
        p = isl_printer_print_str(p, "_");
        p = isl_printer_print_int(p, c0 + 1);
      }
      // io_L2
      for (int c1 = 0; c1 <= 1; c1 += 1)
        if (c1 == 1)
          {
            p = isl_printer_print_str(p, ",");
            p = isl_printer_end_line(p);
            p = isl_printer_start_line(p);
            p = isl_printer_print_str(p, "/* fifo */ ");
            p = isl_printer_print_str(p, "fifo_C_drain_C_drain_IO_L1_out");
            p = isl_printer_print_str(p, "_");
            p = isl_printer_print_int(p, c0);
            p = isl_printer_print_str(p, "_");
            p = isl_printer_print_int(p, c1 + 1);
            p = isl_printer_end_line(p);
            p = isl_printer_indent(p, -4);
            p = isl_printer_start_line(p);
            p = isl_printer_print_str(p, ");");
            p = isl_printer_end_line(p);
            p = isl_printer_start_line(p);
            p = isl_printer_print_str(p, "/* Module Call */");
            p = isl_printer_end_line(p);
            p = isl_printer_end_line(p);
          }
    }

    // array
    {
      {
        p = isl_printer_start_line(p);
        p = isl_printer_print_str(p, "/* Module Call */");
        p = isl_printer_end_line(p);
        // Print calls of module: C_drain_IO_L3_out
        p = isl_printer_start_line(p);
        p = isl_printer_print_str(p, "C_drain_IO_L3_out(");
        p = isl_printer_end_line(p);
        p = isl_printer_indent(p, 4);
        p = isl_printer_start_line(p);
        p = isl_printer_print_str(p, "/* array */ C");
      }
      // io_L3
      for (int c0 = 0; c0 <= 1; c0 += 1)
        if (c0 == 1)
          {
            p = isl_printer_print_str(p, ",");
            p = isl_printer_end_line(p);
            p = isl_printer_start_line(p);
            p = isl_printer_print_str(p, "/* fifo */ ");
            p = isl_printer_print_str(p, "fifo_C_drain_C_drain_IO_L2_out");
            p = isl_printer_print_str(p, "_");
            p = isl_printer_print_int(p, c0 + 1);
            p = isl_printer_end_line(p);
            p = isl_printer_indent(p, -4);
            p = isl_printer_start_line(p);
            p = isl_printer_print_str(p, ");");
            p = isl_printer_end_line(p);
            p = isl_printer_start_line(p);
            p = isl_printer_print_str(p, "/* Module Call */");
            p = isl_printer_end_line(p);
            p = isl_printer_end_line(p);
          }
    }

    p = isl_printer_indent(p, -4);
    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, "}");
    p = isl_printer_end_line(p);
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
