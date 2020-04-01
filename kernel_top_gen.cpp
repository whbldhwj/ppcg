#include <isl/printer.h>
#include "kernel_top_gen.h"
void top_generate(FILE *f)
{
    FILE *fd = fopen("design_info.dat", "w");
    int fifo_cnt;
    isl_ctx *ctx = isl_ctx_alloc();
    isl_printer *p = isl_printer_to_file(ctx, f);

    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, "extern \"C\" {");
    p = isl_printer_end_line(p);
    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, "void kernel0(A_t16 *A, B_t16 *B, C_t4 *C)");
    p = isl_printer_end_line(p);
    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, "{");
    p = isl_printer_end_line(p);
    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, "#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem_A");
    p = isl_printer_end_line(p);
    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, "#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem_B");
    p = isl_printer_end_line(p);
    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, "#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem_C");
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

    fifo_cnt = 0;
    // array
    // io_L3
    for (int c0 = 0; c0 <= 1; c0 += 1) {
      {
        // Count channel number
        fifo_cnt++;
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
        p = isl_printer_print_str(p, " depth=2");
        p = isl_printer_end_line(p);
      }
      if (c0 == 1)
        {
          // Count channel number
          fifo_cnt++;
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
          p = isl_printer_print_str(p, " depth=2");
          p = isl_printer_end_line(p);
        }
    }
    fprintf(fd, "fifo:fifo_A_A_IO_L2_in:%d:16\n", fifo_cnt);

    fifo_cnt = 0;
    // array
    // io_L3
    for (int c0 = 0; c0 <= 1; c0 += 1) {
      {
        // Count channel number
        fifo_cnt++;
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
        p = isl_printer_print_str(p, " depth=2");
        p = isl_printer_end_line(p);
      }
      if (c0 == 1)
        {
          // Count channel number
          fifo_cnt++;
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
          p = isl_printer_print_str(p, " depth=2");
          p = isl_printer_end_line(p);
        }
    }
    fprintf(fd, "fifo:fifo_B_B_IO_L2_in:%d:16\n", fifo_cnt);

    fifo_cnt = 0;
    // array
    for (int c0 = 0; c0 <= 1; c0 += 1)
      for (int c1 = 0; c1 <= 1; c1 += 1) {
        // io_L1
        {
          {
            // Count channel number
            fifo_cnt++;
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
            p = isl_printer_print_str(p, " depth=2");
            p = isl_printer_end_line(p);
          }
          if (c1 == 1)
            {
              // Count channel number
              fifo_cnt++;
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
              p = isl_printer_print_str(p, " depth=2");
              p = isl_printer_end_line(p);
            }
        }
      }
    fprintf(fd, "fifo:fifo_A_PE:%d:8\n", fifo_cnt);

    fifo_cnt = 0;
    // array
    for (int c0 = 0; c0 <= 1; c0 += 1)
      for (int c1 = 0; c1 <= 1; c1 += 1) {
        // io_L1
        {
          {
            // Count channel number
            fifo_cnt++;
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
            p = isl_printer_print_str(p, " depth=2");
            p = isl_printer_end_line(p);
          }
          if (c1 == 1)
            {
              // Count channel number
              fifo_cnt++;
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
              p = isl_printer_print_str(p, " depth=2");
              p = isl_printer_end_line(p);
            }
        }
      }
    fprintf(fd, "fifo:fifo_B_PE:%d:8\n", fifo_cnt);

    fifo_cnt = 0;
    // array
    for (int c0 = 0; c0 <= 1; c0 += 1)
      for (int c1 = 0; c1 <= 1; c1 += 1) {
        // io_L1
        {
          // Count channel number
          fifo_cnt++;
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
          p = isl_printer_print_str(p, " depth=2");
          p = isl_printer_end_line(p);
        }
      }
    fprintf(fd, "fifo:fifo_C_drain_PE:%d:4\n", fifo_cnt);

    fifo_cnt = 0;
    // array
    // io_L3
    for (int c0 = 0; c0 <= 1; c0 += 1) {
      // io_L2
      for (int c1 = 0; c1 <= 1; c1 += 1) {
        {
          // Count channel number
          fifo_cnt++;
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
          p = isl_printer_print_str(p, " depth=2");
          p = isl_printer_end_line(p);
        }
        if (c1 == 1)
          {
            // Count channel number
            fifo_cnt++;
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
            p = isl_printer_print_str(p, " depth=2");
            p = isl_printer_end_line(p);
          }
      }
    }
    fprintf(fd, "fifo:fifo_C_drain_C_drain_IO_L1_out:%d:8\n", fifo_cnt);

    fifo_cnt = 0;
    // array
    // io_L3
    for (int c0 = 0; c0 <= 1; c0 += 1) {
      {
        // Count channel number
        fifo_cnt++;
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
        p = isl_printer_print_str(p, " depth=2");
        p = isl_printer_end_line(p);
      }
      if (c0 == 1)
        {
          // Count channel number
          fifo_cnt++;
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
          p = isl_printer_print_str(p, " depth=2");
          p = isl_printer_end_line(p);
        }
    }
    fprintf(fd, "fifo:fifo_C_drain_C_drain_IO_L2_out:%d:8\n", fifo_cnt);

    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, "/* FIFO Declaration */");
    p = isl_printer_end_line(p);
    p = isl_printer_end_line(p);
    int A_IO_L3_in_cnt = 0;
    int A_IO_L2_in_intra_trans_cnt = 0;
    int A_IO_L2_in_inter_trans_cnt = 0;
    int A_IO_L2_in_inter_trans_boundary_cnt = 0;
    int A_IO_L2_in_cnt = 0;
    int A_IO_L2_in_boundary_cnt = 0;
    int B_IO_L3_in_cnt = 0;
    int B_IO_L2_in_intra_trans_cnt = 0;
    int B_IO_L2_in_inter_trans_cnt = 0;
    int B_IO_L2_in_inter_trans_boundary_cnt = 0;
    int B_IO_L2_in_cnt = 0;
    int B_IO_L2_in_boundary_cnt = 0;
    int PE_cnt = 0;
    int A_PE_dummy_cnt = 0;
    int B_PE_dummy_cnt = 0;
    int C_drain_IO_L1_out_intra_trans_cnt = 0;
    int C_drain_IO_L1_out_inter_trans_cnt = 0;
    int C_drain_IO_L1_out_inter_trans_boundary_cnt = 0;
    int C_drain_IO_L1_out_cnt = 0;
    int C_drain_IO_L1_out_boundary_cnt = 0;
    int C_drain_IO_L2_out_cnt = 0;
    int C_drain_IO_L2_out_boundary_cnt = 0;
    int C_drain_IO_L3_out_cnt = 0;
    // array
    // io_L3
    {
      {
        // Count module number
        A_IO_L3_in_cnt++;
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
      // io_L2
      if (c0 == 0) {
        {
          // Count module number
          A_IO_L2_in_cnt++;
          A_IO_L2_in_intra_trans_cnt++;
          A_IO_L2_in_inter_trans_cnt++;
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
        for (int c1 = 0; c1 <= 1; c1 += 1)
          if (c1 == 0)
            {
              p = isl_printer_print_str(p, ",");
              p = isl_printer_end_line(p);
              p = isl_printer_start_line(p);
              p = isl_printer_print_str(p, "/* fifo */ ");
              p = isl_printer_print_str(p, "fifo_A_PE");
              p = isl_printer_print_str(p, "_");
              p = isl_printer_print_int(p, 0);
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
      } else {
        {
          // Count module number
          A_IO_L2_in_boundary_cnt++;
          A_IO_L2_in_intra_trans_cnt++;
          A_IO_L2_in_inter_trans_boundary_cnt++;
          p = isl_printer_start_line(p);
          p = isl_printer_print_str(p, "/* Module Call */");
          p = isl_printer_end_line(p);
          // Print calls of module: A_IO_L2_in_boundary
          p = isl_printer_start_line(p);
          p = isl_printer_print_str(p, "A_IO_L2_in_boundary(");
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
        }
        for (int c1 = 0; c1 <= 1; c1 += 1)
          if (c1 == 0)
            {
              p = isl_printer_print_str(p, ",");
              p = isl_printer_end_line(p);
              p = isl_printer_start_line(p);
              p = isl_printer_print_str(p, "/* fifo */ ");
              p = isl_printer_print_str(p, "fifo_A_PE");
              p = isl_printer_print_str(p, "_");
              p = isl_printer_print_int(p, 1);
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
    }
    // array
    // io_L3
    {
      {
        // Count module number
        B_IO_L3_in_cnt++;
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
      // io_L2
      if (c0 == 0) {
        {
          // Count module number
          B_IO_L2_in_cnt++;
          B_IO_L2_in_intra_trans_cnt++;
          B_IO_L2_in_inter_trans_cnt++;
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
      } else {
        {
          // Count module number
          B_IO_L2_in_boundary_cnt++;
          B_IO_L2_in_intra_trans_cnt++;
          B_IO_L2_in_inter_trans_boundary_cnt++;
          p = isl_printer_start_line(p);
          p = isl_printer_print_str(p, "/* Module Call */");
          p = isl_printer_end_line(p);
          // Print calls of module: B_IO_L2_in_boundary
          p = isl_printer_start_line(p);
          p = isl_printer_print_str(p, "B_IO_L2_in_boundary(");
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
        }
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
              p = isl_printer_print_int(p, 1);
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
    for (int c0 = 0; c0 <= 1; c0 += 1)
      for (int c1 = 0; c1 <= 1; c1 += 1)
        {
          // Count module number
          PE_cnt++;
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
    // pe_dummy_module
    // array
    for (int c0 = 0; c0 <= 1; c0 += 1)
      for (int c1 = 0; c1 <= 1; c1 += 1)
        if (c1 == 1) {
          // io_L1
          {
            // Count module number
            A_PE_dummy_cnt++;
            p = isl_printer_start_line(p);
            p = isl_printer_print_str(p, "/* Module Call */");
            p = isl_printer_end_line(p);
            // Print calls of module: A_PE_dummy
            p = isl_printer_start_line(p);
            p = isl_printer_print_str(p, "A_PE_dummy(");
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
    // pe_dummy_module
    // array
    for (int c0 = 0; c0 <= 1; c0 += 1)
      for (int c1 = 0; c1 <= 1; c1 += 1)
        if (c1 == 1) {
          // io_L1
          {
            // Count module number
            B_PE_dummy_cnt++;
            p = isl_printer_start_line(p);
            p = isl_printer_print_str(p, "/* Module Call */");
            p = isl_printer_end_line(p);
            // Print calls of module: B_PE_dummy
            p = isl_printer_start_line(p);
            p = isl_printer_print_str(p, "B_PE_dummy(");
            p = isl_printer_end_line(p);
            p = isl_printer_indent(p, 4);
            p = isl_printer_start_line(p);
            p = isl_printer_print_str(p, "/* module id */ ");
            p = isl_printer_print_int(p, c1);
            p = isl_printer_print_str(p, ",");
            p = isl_printer_end_line(p);
            p = isl_printer_start_line(p);
            p = isl_printer_print_str(p, "/* module id */ ");
            p = isl_printer_print_int(p, c0);
            p = isl_printer_print_str(p, ",");
            p = isl_printer_end_line(p);
            p = isl_printer_start_line(p);
            p = isl_printer_print_str(p, "/* fifo */ ");
            p = isl_printer_print_str(p, "fifo_B_PE");
            p = isl_printer_print_str(p, "_");
            p = isl_printer_print_int(p, c1 + 1);
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
      // io_L2
      for (int c1 = 0; c1 <= 1; c1 += 1) {
        // io_L1
        if (c1 == 0) {
          {
            // Count module number
            C_drain_IO_L1_out_cnt++;
            C_drain_IO_L1_out_intra_trans_cnt++;
            C_drain_IO_L1_out_inter_trans_cnt++;
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
            p = isl_printer_print_int(p, c1 + 1);
            p = isl_printer_print_str(p, ",");
            p = isl_printer_end_line(p);
            p = isl_printer_start_line(p);
            p = isl_printer_print_str(p, "/* fifo */ ");
            p = isl_printer_print_str(p, "fifo_C_drain_C_drain_IO_L1_out");
            p = isl_printer_print_str(p, "_");
            p = isl_printer_print_int(p, c0);
            p = isl_printer_print_str(p, "_");
            p = isl_printer_print_int(p, c1);
          }
          {
            p = isl_printer_print_str(p, ",");
            p = isl_printer_end_line(p);
            p = isl_printer_start_line(p);
            p = isl_printer_print_str(p, "/* fifo */ ");
            p = isl_printer_print_str(p, "fifo_C_drain_PE");
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
        } else {
          {
            // Count module number
            C_drain_IO_L1_out_boundary_cnt++;
            C_drain_IO_L1_out_intra_trans_cnt++;
            C_drain_IO_L1_out_inter_trans_boundary_cnt++;
            p = isl_printer_start_line(p);
            p = isl_printer_print_str(p, "/* Module Call */");
            p = isl_printer_end_line(p);
            // Print calls of module: C_drain_IO_L1_out_boundary
            p = isl_printer_start_line(p);
            p = isl_printer_print_str(p, "C_drain_IO_L1_out_boundary(");
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
          }
          {
            p = isl_printer_print_str(p, ",");
            p = isl_printer_end_line(p);
            p = isl_printer_start_line(p);
            p = isl_printer_print_str(p, "/* fifo */ ");
            p = isl_printer_print_str(p, "fifo_C_drain_PE");
            p = isl_printer_print_str(p, "_");
            p = isl_printer_print_int(p, 1);
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
    }
    // array
    // io_L3
    for (int c0 = 0; c0 <= 1; c0 += 1) {
      // io_L2
      if (c0 == 0) {
        {
          // Count module number
          C_drain_IO_L2_out_cnt++;
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
          p = isl_printer_print_int(p, c0 + 1);
          p = isl_printer_print_str(p, ",");
          p = isl_printer_end_line(p);
          p = isl_printer_start_line(p);
          p = isl_printer_print_str(p, "/* fifo */ ");
          p = isl_printer_print_str(p, "fifo_C_drain_C_drain_IO_L2_out");
          p = isl_printer_print_str(p, "_");
          p = isl_printer_print_int(p, c0);
        }
        for (int c1 = 0; c1 <= 1; c1 += 1)
          if (c1 == 0)
            {
              p = isl_printer_print_str(p, ",");
              p = isl_printer_end_line(p);
              p = isl_printer_start_line(p);
              p = isl_printer_print_str(p, "/* fifo */ ");
              p = isl_printer_print_str(p, "fifo_C_drain_C_drain_IO_L1_out");
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
      } else {
        {
          // Count module number
          C_drain_IO_L2_out_boundary_cnt++;
          p = isl_printer_start_line(p);
          p = isl_printer_print_str(p, "/* Module Call */");
          p = isl_printer_end_line(p);
          // Print calls of module: C_drain_IO_L2_out_boundary
          p = isl_printer_start_line(p);
          p = isl_printer_print_str(p, "C_drain_IO_L2_out_boundary(");
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
        }
        for (int c1 = 0; c1 <= 1; c1 += 1)
          if (c1 == 0)
            {
              p = isl_printer_print_str(p, ",");
              p = isl_printer_end_line(p);
              p = isl_printer_start_line(p);
              p = isl_printer_print_str(p, "/* fifo */ ");
              p = isl_printer_print_str(p, "fifo_C_drain_C_drain_IO_L1_out");
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
      }
    }
    // array
    // io_L3
    {
      {
        // Count module number
        C_drain_IO_L3_out_cnt++;
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
      for (int c0 = 0; c0 <= 1; c0 += 1)
        if (c0 == 0)
          {
            p = isl_printer_print_str(p, ",");
            p = isl_printer_end_line(p);
            p = isl_printer_start_line(p);
            p = isl_printer_print_str(p, "/* fifo */ ");
            p = isl_printer_print_str(p, "fifo_C_drain_C_drain_IO_L2_out");
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
    fprintf(fd, "module:A_IO_L3_in:%d\n", A_IO_L3_in_cnt);
    fprintf(fd, "module:A_IO_L2_in_intra_trans:%d\n", A_IO_L2_in_intra_trans_cnt);
    fprintf(fd, "module:A_IO_L2_in_inter_trans:%d\n", A_IO_L2_in_inter_trans_cnt);
    fprintf(fd, "module:A_IO_L2_in_inter_trans_boundary:%d\n", A_IO_L2_in_inter_trans_boundary_cnt);
    fprintf(fd, "module:A_IO_L2_in:%d\n", A_IO_L2_in_cnt);
    fprintf(fd, "module:A_IO_L2_in_boundary:%d\n", A_IO_L2_in_boundary_cnt);
    fprintf(fd, "module:B_IO_L3_in:%d\n", B_IO_L3_in_cnt);
    fprintf(fd, "module:B_IO_L2_in_intra_trans:%d\n", B_IO_L2_in_intra_trans_cnt);
    fprintf(fd, "module:B_IO_L2_in_inter_trans:%d\n", B_IO_L2_in_inter_trans_cnt);
    fprintf(fd, "module:B_IO_L2_in_inter_trans_boundary:%d\n", B_IO_L2_in_inter_trans_boundary_cnt);
    fprintf(fd, "module:B_IO_L2_in:%d\n", B_IO_L2_in_cnt);
    fprintf(fd, "module:B_IO_L2_in_boundary:%d\n", B_IO_L2_in_boundary_cnt);
    fprintf(fd, "module:PE:%d\n", PE_cnt);
    fprintf(fd, "module:A_PE_dummy:%d\n", A_PE_dummy_cnt);
    fprintf(fd, "module:B_PE_dummy:%d\n", B_PE_dummy_cnt);
    fprintf(fd, "module:C_drain_IO_L1_out_intra_trans:%d\n", C_drain_IO_L1_out_intra_trans_cnt);
    fprintf(fd, "module:C_drain_IO_L1_out_inter_trans:%d\n", C_drain_IO_L1_out_inter_trans_cnt);
    fprintf(fd, "module:C_drain_IO_L1_out_inter_trans_boundary:%d\n", C_drain_IO_L1_out_inter_trans_boundary_cnt);
    fprintf(fd, "module:C_drain_IO_L1_out:%d\n", C_drain_IO_L1_out_cnt);
    fprintf(fd, "module:C_drain_IO_L1_out_boundary:%d\n", C_drain_IO_L1_out_boundary_cnt);
    fprintf(fd, "module:C_drain_IO_L2_out:%d\n", C_drain_IO_L2_out_cnt);
    fprintf(fd, "module:C_drain_IO_L2_out_boundary:%d\n", C_drain_IO_L2_out_boundary_cnt);
    fprintf(fd, "module:C_drain_IO_L3_out:%d\n", C_drain_IO_L3_out_cnt);

    p = isl_printer_indent(p, -4);
    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, "}");
    p = isl_printer_end_line(p);
    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, "}");
    p = isl_printer_end_line(p);

    fclose(fd);
    isl_printer_free(p);
    isl_ctx_free(ctx);
}

int main()
{
  FILE *f = fopen("temp.c", "w");
  top_generate(f);
}
