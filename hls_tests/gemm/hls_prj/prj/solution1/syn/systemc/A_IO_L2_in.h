// ==============================================================
// RTL generated by Vivado(TM) HLS - High-Level Synthesis from C, C++ and OpenCL
// Version: 2019.2
// Copyright (C) 1986-2019 Xilinx, Inc. All Rights Reserved.
// 
// ===========================================================

#ifndef _A_IO_L2_in_HH_
#define _A_IO_L2_in_HH_

#include "systemc.h"
#include "AESL_pkg.h"

#include "A_IO_L2_in_intra_tra.h"
#include "A_IO_L2_in_inter_tra_1.h"
#include "A_IO_L2_in_local_bkb.h"

namespace ap_rtl {

struct A_IO_L2_in : public sc_module {
    // Port declarations 19
    sc_in_clk ap_clk;
    sc_in< sc_logic > ap_rst;
    sc_in< sc_logic > ap_start;
    sc_in< sc_logic > start_full_n;
    sc_out< sc_logic > ap_done;
    sc_in< sc_logic > ap_continue;
    sc_out< sc_logic > ap_idle;
    sc_out< sc_logic > ap_ready;
    sc_out< sc_logic > start_out;
    sc_out< sc_logic > start_write;
    sc_in< sc_lv<128> > fifo_A_in_V_V_dout;
    sc_in< sc_logic > fifo_A_in_V_V_empty_n;
    sc_out< sc_logic > fifo_A_in_V_V_read;
    sc_out< sc_lv<128> > fifo_A_out_V_V_din;
    sc_in< sc_logic > fifo_A_out_V_V_full_n;
    sc_out< sc_logic > fifo_A_out_V_V_write;
    sc_out< sc_lv<64> > fifo_A_local_out_V_V_din;
    sc_in< sc_logic > fifo_A_local_out_V_V_full_n;
    sc_out< sc_logic > fifo_A_local_out_V_V_write;


    // Module declarations
    A_IO_L2_in(sc_module_name name);
    SC_HAS_PROCESS(A_IO_L2_in);

    ~A_IO_L2_in();

    sc_trace_file* mVcdFile;

    A_IO_L2_in_local_bkb* local_A_ping_0_V_U;
    A_IO_L2_in_local_bkb* local_A_pong_0_V_U;
    A_IO_L2_in_intra_tra* grp_A_IO_L2_in_intra_tra_fu_131;
    A_IO_L2_in_inter_tra_1* grp_A_IO_L2_in_inter_tra_1_fu_139;
    sc_signal< sc_logic > real_start;
    sc_signal< sc_logic > start_once_reg;
    sc_signal< sc_logic > ap_done_reg;
    sc_signal< sc_lv<6> > ap_CS_fsm;
    sc_signal< sc_logic > ap_CS_fsm_state1;
    sc_signal< sc_logic > internal_ap_ready;
    sc_signal< sc_lv<2> > c0_fu_163_p2;
    sc_signal< sc_lv<2> > c0_reg_214;
    sc_signal< sc_logic > ap_CS_fsm_state2;
    sc_signal< sc_lv<1> > intra_trans_en_0_loa_4_reg_219;
    sc_signal< sc_lv<1> > icmp_ln130_fu_157_p2;
    sc_signal< sc_lv<2> > c1_fu_175_p2;
    sc_signal< sc_lv<2> > c1_reg_227;
    sc_signal< sc_logic > ap_CS_fsm_state3;
    sc_signal< sc_lv<2> > c2_fu_187_p2;
    sc_signal< sc_lv<2> > c2_reg_235;
    sc_signal< sc_logic > ap_CS_fsm_state4;
    sc_signal< sc_lv<1> > intra_trans_en_0_loa_reg_240;
    sc_signal< sc_lv<1> > icmp_ln132_fu_181_p2;
    sc_signal< sc_lv<1> > xor_ln151_fu_193_p2;
    sc_signal< sc_logic > ap_CS_fsm_state5;
    sc_signal< sc_logic > grp_A_IO_L2_in_inter_tra_1_fu_139_ap_ready;
    sc_signal< sc_logic > grp_A_IO_L2_in_inter_tra_1_fu_139_ap_done;
    sc_signal< sc_lv<1> > arb_2_reg_108;
    sc_signal< sc_logic > grp_A_IO_L2_in_intra_tra_fu_131_ap_ready;
    sc_signal< sc_logic > grp_A_IO_L2_in_intra_tra_fu_131_ap_done;
    sc_signal< bool > ap_block_state5_on_subcall_done;
    sc_signal< sc_lv<1> > local_A_ping_0_V_address0;
    sc_signal< sc_logic > local_A_ping_0_V_ce0;
    sc_signal< sc_logic > local_A_ping_0_V_we0;
    sc_signal< sc_lv<128> > local_A_ping_0_V_q0;
    sc_signal< sc_lv<1> > local_A_pong_0_V_address0;
    sc_signal< sc_logic > local_A_pong_0_V_ce0;
    sc_signal< sc_logic > local_A_pong_0_V_we0;
    sc_signal< sc_lv<128> > local_A_pong_0_V_q0;
    sc_signal< sc_logic > grp_A_IO_L2_in_intra_tra_fu_131_ap_start;
    sc_signal< sc_logic > grp_A_IO_L2_in_intra_tra_fu_131_ap_idle;
    sc_signal< sc_lv<1> > grp_A_IO_L2_in_intra_tra_fu_131_local_A_0_V_address0;
    sc_signal< sc_logic > grp_A_IO_L2_in_intra_tra_fu_131_local_A_0_V_ce0;
    sc_signal< sc_lv<128> > grp_A_IO_L2_in_intra_tra_fu_131_local_A_0_V_q0;
    sc_signal< sc_lv<64> > grp_A_IO_L2_in_intra_tra_fu_131_fifo_A_local_out_V_V_din;
    sc_signal< sc_logic > grp_A_IO_L2_in_intra_tra_fu_131_fifo_A_local_out_V_V_write;
    sc_signal< sc_logic > grp_A_IO_L2_in_intra_tra_fu_131_en;
    sc_signal< sc_logic > grp_A_IO_L2_in_inter_tra_1_fu_139_ap_start;
    sc_signal< sc_logic > grp_A_IO_L2_in_inter_tra_1_fu_139_ap_idle;
    sc_signal< sc_lv<1> > grp_A_IO_L2_in_inter_tra_1_fu_139_local_A_0_V_address0;
    sc_signal< sc_logic > grp_A_IO_L2_in_inter_tra_1_fu_139_local_A_0_V_ce0;
    sc_signal< sc_logic > grp_A_IO_L2_in_inter_tra_1_fu_139_local_A_0_V_we0;
    sc_signal< sc_lv<128> > grp_A_IO_L2_in_inter_tra_1_fu_139_local_A_0_V_d0;
    sc_signal< sc_logic > grp_A_IO_L2_in_inter_tra_1_fu_139_fifo_A_in_V_V_read;
    sc_signal< sc_lv<128> > grp_A_IO_L2_in_inter_tra_1_fu_139_fifo_A_out_V_V_din;
    sc_signal< sc_logic > grp_A_IO_L2_in_inter_tra_1_fu_139_fifo_A_out_V_V_write;
    sc_signal< sc_lv<2> > c0_prev_reg_86;
    sc_signal< bool > ap_block_state1;
    sc_signal< sc_lv<1> > icmp_ln131_fu_169_p2;
    sc_signal< sc_lv<2> > c1_prev_reg_97;
    sc_signal< sc_lv<1> > ap_phi_mux_arb_2_phi_fu_112_p4;
    sc_signal< sc_lv<2> > c2_prev_reg_120;
    sc_signal< sc_logic > grp_A_IO_L2_in_intra_tra_fu_131_ap_start_reg;
    sc_signal< sc_logic > ap_CS_fsm_state6;
    sc_signal< sc_logic > grp_A_IO_L2_in_inter_tra_1_fu_139_ap_start_reg;
    sc_signal< sc_lv<1> > intra_trans_en_0_fu_74;
    sc_signal< sc_lv<6> > ap_NS_fsm;
    static const sc_logic ap_const_logic_1;
    static const sc_logic ap_const_logic_0;
    static const sc_lv<6> ap_ST_fsm_state1;
    static const sc_lv<6> ap_ST_fsm_state2;
    static const sc_lv<6> ap_ST_fsm_state3;
    static const sc_lv<6> ap_ST_fsm_state4;
    static const sc_lv<6> ap_ST_fsm_state5;
    static const sc_lv<6> ap_ST_fsm_state6;
    static const sc_lv<32> ap_const_lv32_0;
    static const sc_lv<32> ap_const_lv32_1;
    static const sc_lv<1> ap_const_lv1_1;
    static const sc_lv<32> ap_const_lv32_2;
    static const sc_lv<32> ap_const_lv32_3;
    static const sc_lv<1> ap_const_lv1_0;
    static const sc_lv<32> ap_const_lv32_4;
    static const bool ap_const_boolean_0;
    static const sc_lv<2> ap_const_lv2_0;
    static const sc_lv<32> ap_const_lv32_5;
    static const sc_lv<2> ap_const_lv2_2;
    static const sc_lv<2> ap_const_lv2_1;
    static const bool ap_const_boolean_1;
    // Thread declarations
    void thread_ap_clk_no_reset_();
    void thread_ap_CS_fsm_state1();
    void thread_ap_CS_fsm_state2();
    void thread_ap_CS_fsm_state3();
    void thread_ap_CS_fsm_state4();
    void thread_ap_CS_fsm_state5();
    void thread_ap_CS_fsm_state6();
    void thread_ap_block_state1();
    void thread_ap_block_state5_on_subcall_done();
    void thread_ap_done();
    void thread_ap_idle();
    void thread_ap_phi_mux_arb_2_phi_fu_112_p4();
    void thread_ap_ready();
    void thread_c0_fu_163_p2();
    void thread_c1_fu_175_p2();
    void thread_c2_fu_187_p2();
    void thread_fifo_A_in_V_V_read();
    void thread_fifo_A_local_out_V_V_din();
    void thread_fifo_A_local_out_V_V_write();
    void thread_fifo_A_out_V_V_din();
    void thread_fifo_A_out_V_V_write();
    void thread_grp_A_IO_L2_in_inter_tra_1_fu_139_ap_start();
    void thread_grp_A_IO_L2_in_intra_tra_fu_131_ap_start();
    void thread_grp_A_IO_L2_in_intra_tra_fu_131_en();
    void thread_grp_A_IO_L2_in_intra_tra_fu_131_local_A_0_V_q0();
    void thread_icmp_ln130_fu_157_p2();
    void thread_icmp_ln131_fu_169_p2();
    void thread_icmp_ln132_fu_181_p2();
    void thread_internal_ap_ready();
    void thread_local_A_ping_0_V_address0();
    void thread_local_A_ping_0_V_ce0();
    void thread_local_A_ping_0_V_we0();
    void thread_local_A_pong_0_V_address0();
    void thread_local_A_pong_0_V_ce0();
    void thread_local_A_pong_0_V_we0();
    void thread_real_start();
    void thread_start_out();
    void thread_start_write();
    void thread_xor_ln151_fu_193_p2();
    void thread_ap_NS_fsm();
};

}

using namespace ap_rtl;

#endif
