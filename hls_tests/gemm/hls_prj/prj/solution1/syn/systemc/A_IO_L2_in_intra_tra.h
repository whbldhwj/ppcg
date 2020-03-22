// ==============================================================
// RTL generated by Vivado(TM) HLS - High-Level Synthesis from C, C++ and OpenCL
// Version: 2019.2
// Copyright (C) 1986-2019 Xilinx, Inc. All Rights Reserved.
// 
// ===========================================================

#ifndef _A_IO_L2_in_intra_tra_HH_
#define _A_IO_L2_in_intra_tra_HH_

#include "systemc.h"
#include "AESL_pkg.h"


namespace ap_rtl {

struct A_IO_L2_in_intra_tra : public sc_module {
    // Port declarations 13
    sc_in_clk ap_clk;
    sc_in< sc_logic > ap_rst;
    sc_in< sc_logic > ap_start;
    sc_out< sc_logic > ap_done;
    sc_out< sc_logic > ap_idle;
    sc_out< sc_logic > ap_ready;
    sc_out< sc_lv<1> > local_A_0_V_address0;
    sc_out< sc_logic > local_A_0_V_ce0;
    sc_in< sc_lv<128> > local_A_0_V_q0;
    sc_out< sc_lv<64> > fifo_A_local_out_V_V_din;
    sc_in< sc_logic > fifo_A_local_out_V_V_full_n;
    sc_out< sc_logic > fifo_A_local_out_V_V_write;
    sc_in< sc_logic > en;


    // Module declarations
    A_IO_L2_in_intra_tra(sc_module_name name);
    SC_HAS_PROCESS(A_IO_L2_in_intra_tra);

    ~A_IO_L2_in_intra_tra();

    sc_trace_file* mVcdFile;

    sc_signal< sc_lv<3> > ap_CS_fsm;
    sc_signal< sc_logic > ap_CS_fsm_state1;
    sc_signal< sc_logic > fifo_A_local_out_V_V_blk_n;
    sc_signal< sc_logic > ap_CS_fsm_pp0_stage0;
    sc_signal< sc_logic > ap_enable_reg_pp0_iter1;
    sc_signal< bool > ap_block_pp0_stage0;
    sc_signal< sc_lv<1> > icmp_ln38_reg_261;
    sc_signal< sc_lv<4> > indvar_flatten11_reg_98;
    sc_signal< sc_lv<2> > c5_0_reg_109;
    sc_signal< sc_lv<4> > indvar_flatten_reg_120;
    sc_signal< sc_lv<2> > c7_0_reg_131;
    sc_signal< sc_lv<1> > en_read_read_fu_72_p2;
    sc_signal< sc_lv<1> > icmp_ln38_fu_142_p2;
    sc_signal< bool > ap_block_state2_pp0_stage0_iter0;
    sc_signal< bool > ap_block_state3_pp0_stage0_iter1;
    sc_signal< bool > ap_block_pp0_stage0_11001;
    sc_signal< sc_lv<4> > add_ln38_fu_148_p2;
    sc_signal< sc_logic > ap_enable_reg_pp0_iter0;
    sc_signal< sc_lv<2> > select_ln321_fu_166_p3;
    sc_signal< sc_lv<2> > select_ln321_reg_270;
    sc_signal< sc_lv<1> > trunc_ln321_fu_174_p1;
    sc_signal< sc_lv<1> > trunc_ln321_reg_275;
    sc_signal< sc_lv<2> > c7_fu_215_p2;
    sc_signal< sc_lv<4> > select_ln40_fu_227_p3;
    sc_signal< bool > ap_block_pp0_stage0_subdone;
    sc_signal< sc_logic > ap_condition_pp0_exit_iter0_state2;
    sc_signal< sc_lv<2> > ap_phi_mux_c5_0_phi_fu_113_p4;
    sc_signal< sc_lv<64> > zext_ln50_fu_210_p1;
    sc_signal< bool > ap_block_pp0_stage0_01001;
    sc_signal< sc_lv<1> > icmp_ln40_fu_160_p2;
    sc_signal< sc_lv<2> > c5_fu_154_p2;
    sc_signal< sc_lv<1> > icmp_ln42_fu_184_p2;
    sc_signal< sc_lv<1> > xor_ln321_fu_178_p2;
    sc_signal< sc_lv<1> > and_ln321_fu_190_p2;
    sc_signal< sc_lv<1> > or_ln42_fu_196_p2;
    sc_signal< sc_lv<2> > select_ln42_fu_202_p3;
    sc_signal< sc_lv<4> > add_ln40_fu_221_p2;
    sc_signal< sc_lv<64> > buf_data_split_1_V_fu_239_p4;
    sc_signal< sc_lv<64> > buf_data_split_0_V_fu_235_p1;
    sc_signal< sc_logic > ap_CS_fsm_state4;
    sc_signal< sc_lv<3> > ap_NS_fsm;
    sc_signal< sc_logic > ap_idle_pp0;
    sc_signal< sc_logic > ap_enable_pp0;
    static const sc_logic ap_const_logic_1;
    static const sc_logic ap_const_logic_0;
    static const sc_lv<3> ap_ST_fsm_state1;
    static const sc_lv<3> ap_ST_fsm_pp0_stage0;
    static const sc_lv<3> ap_ST_fsm_state4;
    static const bool ap_const_boolean_1;
    static const sc_lv<32> ap_const_lv32_0;
    static const sc_lv<32> ap_const_lv32_1;
    static const bool ap_const_boolean_0;
    static const sc_lv<1> ap_const_lv1_0;
    static const sc_lv<1> ap_const_lv1_1;
    static const sc_lv<4> ap_const_lv4_0;
    static const sc_lv<2> ap_const_lv2_0;
    static const sc_lv<4> ap_const_lv4_8;
    static const sc_lv<4> ap_const_lv4_1;
    static const sc_lv<2> ap_const_lv2_1;
    static const sc_lv<4> ap_const_lv4_4;
    static const sc_lv<2> ap_const_lv2_2;
    static const sc_lv<32> ap_const_lv32_40;
    static const sc_lv<32> ap_const_lv32_7F;
    static const sc_lv<32> ap_const_lv32_2;
    // Thread declarations
    void thread_ap_clk_no_reset_();
    void thread_add_ln38_fu_148_p2();
    void thread_add_ln40_fu_221_p2();
    void thread_and_ln321_fu_190_p2();
    void thread_ap_CS_fsm_pp0_stage0();
    void thread_ap_CS_fsm_state1();
    void thread_ap_CS_fsm_state4();
    void thread_ap_block_pp0_stage0();
    void thread_ap_block_pp0_stage0_01001();
    void thread_ap_block_pp0_stage0_11001();
    void thread_ap_block_pp0_stage0_subdone();
    void thread_ap_block_state2_pp0_stage0_iter0();
    void thread_ap_block_state3_pp0_stage0_iter1();
    void thread_ap_condition_pp0_exit_iter0_state2();
    void thread_ap_done();
    void thread_ap_enable_pp0();
    void thread_ap_idle();
    void thread_ap_idle_pp0();
    void thread_ap_phi_mux_c5_0_phi_fu_113_p4();
    void thread_ap_ready();
    void thread_buf_data_split_0_V_fu_235_p1();
    void thread_buf_data_split_1_V_fu_239_p4();
    void thread_c5_fu_154_p2();
    void thread_c7_fu_215_p2();
    void thread_en_read_read_fu_72_p2();
    void thread_fifo_A_local_out_V_V_blk_n();
    void thread_fifo_A_local_out_V_V_din();
    void thread_fifo_A_local_out_V_V_write();
    void thread_icmp_ln38_fu_142_p2();
    void thread_icmp_ln40_fu_160_p2();
    void thread_icmp_ln42_fu_184_p2();
    void thread_local_A_0_V_address0();
    void thread_local_A_0_V_ce0();
    void thread_or_ln42_fu_196_p2();
    void thread_select_ln321_fu_166_p3();
    void thread_select_ln40_fu_227_p3();
    void thread_select_ln42_fu_202_p3();
    void thread_trunc_ln321_fu_174_p1();
    void thread_xor_ln321_fu_178_p2();
    void thread_zext_ln50_fu_210_p1();
    void thread_ap_NS_fsm();
};

}

using namespace ap_rtl;

#endif
