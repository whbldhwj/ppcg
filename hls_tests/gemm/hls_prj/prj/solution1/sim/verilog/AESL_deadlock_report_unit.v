`timescale 1 ns / 1 ps

module AESL_deadlock_report_unit #( parameter PROC_NUM = 4 ) (
    input reset,
    input clock,
    input [PROC_NUM - 1:0] dl_in_vec,
    output dl_detect_out,
    output reg [PROC_NUM - 1:0] origin,
    output token_clear);
   
    // FSM states
    localparam ST_IDLE = 2'b0;
    localparam ST_DL_DETECTED = 2'b1;
    localparam ST_DL_REPORT = 2'b10;

    reg [1:0] CS_fsm;
    reg [1:0] NS_fsm;
    reg [PROC_NUM - 1:0] dl_detect_reg;
    reg [PROC_NUM - 1:0] dl_done_reg;
    reg [PROC_NUM - 1:0] origin_reg;
    reg [PROC_NUM - 1:0] dl_in_vec_reg;
    integer i;
    integer fp;

    // FSM State machine
    always @ (negedge reset or posedge clock) begin
        if (~reset) begin
            CS_fsm <= ST_IDLE;
        end
        else begin
            CS_fsm <= NS_fsm;
        end
    end
    always @ (CS_fsm or dl_in_vec or dl_detect_reg or dl_done_reg or dl_in_vec or origin_reg) begin
        NS_fsm = CS_fsm;
        case (CS_fsm)
            ST_IDLE : begin
                if (|dl_in_vec) begin
                    NS_fsm = ST_DL_DETECTED;
                end
            end
            ST_DL_DETECTED: begin
                // has unreported deadlock cycle
                if (dl_detect_reg != dl_done_reg) begin
                    NS_fsm = ST_DL_REPORT;
                end
            end
            ST_DL_REPORT: begin
                if (|(dl_in_vec & origin_reg)) begin
                    NS_fsm = ST_DL_DETECTED;
                end
            end
        endcase
    end

    // dl_detect_reg record the procs that first detect deadlock
    always @ (negedge reset or posedge clock) begin
        if (~reset) begin
            dl_detect_reg <= 'b0;
        end
        else begin
            if (CS_fsm == ST_IDLE) begin
                dl_detect_reg <= dl_in_vec;
            end
        end
    end

    // dl_detect_out keeps in high after deadlock detected
    assign dl_detect_out = |dl_detect_reg;

    // dl_done_reg record the cycles has been reported
    always @ (negedge reset or posedge clock) begin
        if (~reset) begin
            dl_done_reg <= 'b0;
        end
        else begin
            if ((CS_fsm == ST_DL_REPORT) && (|(dl_in_vec & dl_detect_reg) == 'b1)) begin
                dl_done_reg <= dl_done_reg | dl_in_vec;
            end
        end
    end

    // clear token once a cycle is done
    assign token_clear = (CS_fsm == ST_DL_REPORT) ? ((|(dl_in_vec & origin_reg)) ? 'b1 : 'b0) : 'b0;

    // origin_reg record the current cycle start id
    always @ (negedge reset or posedge clock) begin
        if (~reset) begin
            origin_reg <= 'b0;
        end
        else begin
            if (CS_fsm == ST_DL_DETECTED) begin
                origin_reg <= origin;
            end
        end
    end
   
    // origin will be valid for only one cycle
    always @ (CS_fsm or dl_detect_reg or dl_done_reg) begin
        origin = 'b0;
        if (CS_fsm == ST_DL_DETECTED) begin
            for (i = 0; i < PROC_NUM; i = i + 1) begin
                if (dl_detect_reg[i] & ~dl_done_reg[i] & ~(|origin)) begin
                    origin = 'b1 << i;
                end
            end
        end
    end
    
    // dl_in_vec_reg record the current cycle dl_in_vec
    always @ (negedge reset or posedge clock) begin
        if (~reset) begin
            dl_in_vec_reg <= 'b0;
        end
        else begin
            if (CS_fsm == ST_DL_DETECTED) begin
                dl_in_vec_reg <= origin;
            end
            else if (CS_fsm == ST_DL_REPORT) begin
                dl_in_vec_reg <= dl_in_vec;
            end
        end
    end
    
    // get the first valid proc index in dl vector
    function integer proc_index(input [PROC_NUM - 1:0] dl_vec);
        begin
            proc_index = 0;
            for (i = 0; i < PROC_NUM; i = i + 1) begin
                if (dl_vec[i]) begin
                    proc_index = i;
                end
            end
        end
    endfunction

    // get the proc path based on dl vector
    function [344:0] proc_path(input [PROC_NUM - 1:0] dl_vec);
        integer index;
        begin
            index = proc_index(dl_vec);
            case (index)
                0 : begin
                    proc_path = "kernel0.kernel0_entry6_U0";
                end
                1 : begin
                    proc_path = "kernel0.A_IO_L3_in_U0";
                end
                2 : begin
                    proc_path = "kernel0.A_IO_L2_in_U0";
                end
                3 : begin
                    proc_path = "kernel0.A_IO_L2_in_tail_U0";
                end
                4 : begin
                    proc_path = "kernel0.B_IO_L3_in_U0";
                end
                5 : begin
                    proc_path = "kernel0.B_IO_L2_in_U0";
                end
                6 : begin
                    proc_path = "kernel0.B_IO_L2_in_tail_U0";
                end
                7 : begin
                    proc_path = "kernel0.PE139_U0";
                end
                8 : begin
                    proc_path = "kernel0.PE140_U0";
                end
                9 : begin
                    proc_path = "kernel0.PE_A_dummy141_U0";
                end
                10 : begin
                    proc_path = "kernel0.PE142_U0";
                end
                11 : begin
                    proc_path = "kernel0.PE_B_dummy143_U0";
                end
                12 : begin
                    proc_path = "kernel0.PE_U0";
                end
                13 : begin
                    proc_path = "kernel0.PE_A_dummy_U0";
                end
                14 : begin
                    proc_path = "kernel0.PE_B_dummy_U0";
                end
                15 : begin
                    proc_path = "kernel0.C_drain_IO_L1_out_he_U0";
                end
                16 : begin
                    proc_path = "kernel0.C_drain_IO_L1_out145_U0";
                end
                17 : begin
                    proc_path = "kernel0.C_drain_IO_L1_out_he_1_U0";
                end
                18 : begin
                    proc_path = "kernel0.C_drain_IO_L1_out_U0";
                end
                19 : begin
                    proc_path = "kernel0.C_drain_IO_L2_out_he_U0";
                end
                20 : begin
                    proc_path = "kernel0.C_drain_IO_L2_out_U0";
                end
                21 : begin
                    proc_path = "kernel0.C_drain_IO_L3_out_U0";
                end
                default : begin
                    proc_path = "unknown";
                end
            endcase
        end
    endfunction

    // print the headlines of deadlock detection
    task print_dl_head;
        begin
            $display("\n//////////////////////////////////////////////////////////////////////////////");
            $display("// ERROR!!! DEADLOCK DETECTED at %0t ns! SIMULATION WILL BE STOPPED! //", $time);
            $display("//////////////////////////////////////////////////////////////////////////////");
            fp = $fopen("deadlock_db.dat", "w");
        end
    endtask

    // print the start of a cycle
    task print_cycle_start(input reg [344:0] proc_path, input integer cycle_id);
        begin
            $display("/////////////////////////");
            $display("// Dependence cycle %0d:", cycle_id);
            $display("// (1): Process: %0s", proc_path);
            $fdisplay(fp, "Dependence_Cycle_ID %0d", cycle_id);
            $fdisplay(fp, "Dependence_Process_ID 1");
            $fdisplay(fp, "Dependence_Process_path %0s", proc_path);
        end
    endtask

    // print the end of deadlock detection
    task print_dl_end(input integer num);
        begin
            $display("////////////////////////////////////////////////////////////////////////");
            $display("// Totally %0d cycles detected!", num);
            $display("////////////////////////////////////////////////////////////////////////");
            $fdisplay(fp, "Dependence_Cycle_Number %0d", num);
            $fclose(fp);
        end
    endtask

    // print one proc component in the cycle
    task print_cycle_proc_comp(input reg [344:0] proc_path, input integer cycle_comp_id);
        begin
            $display("// (%0d): Process: %0s", cycle_comp_id, proc_path);
            $fdisplay(fp, "Dependence_Process_ID %0d", cycle_comp_id);
            $fdisplay(fp, "Dependence_Process_path %0s", proc_path);
        end
    endtask

    // print one channel component in the cycle
    task print_cycle_chan_comp(input [PROC_NUM - 1:0] dl_vec1, input [PROC_NUM - 1:0] dl_vec2);
        reg [344:0] chan_path;
        integer index1;
        integer index2;
        begin
            index1 = proc_index(dl_vec1);
            index2 = proc_index(dl_vec2);
            case (index1)
                0 : begin
                    case(index2)
                    1: begin
                        if (~AESL_inst_kernel0.kernel0_entry6_U0.A_V_out_blk_n) begin
                            chan_path = "kernel0.A_V_c_U";
                            if (~AESL_inst_kernel0.A_V_c_U.if_empty_n) begin
                                $display("//      Channel: %0s, EMPTY", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~AESL_inst_kernel0.A_V_c_U.if_full_n) begin
                                $display("//      Channel: %0s, FULL", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                            else begin
                                $display("//      Channel: %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status NULL");
                            end
                        end
                        if (((AESL_inst_kernel0.kernel0_entry6_U0_ap_ready_count[0]) & AESL_inst_kernel0.kernel0_entry6_U0.ap_idle & ~(AESL_inst_kernel0.A_IO_L3_in_U0_ap_ready_count[0]))) begin
                            chan_path = "";
                            if (((AESL_inst_kernel0.kernel0_entry6_U0_ap_ready_count[0]) & AESL_inst_kernel0.kernel0_entry6_U0.ap_idle & ~(AESL_inst_kernel0.A_IO_L3_in_U0_ap_ready_count[0]))) begin
                                $display("//      Deadlocked by sync logic between input processes");
                                $display("//      Please increase channel depth");
                            end
                        end
                    end
                    4: begin
                        if (~AESL_inst_kernel0.kernel0_entry6_U0.B_V_out_blk_n) begin
                            chan_path = "kernel0.B_V_c_U";
                            if (~AESL_inst_kernel0.B_V_c_U.if_empty_n) begin
                                $display("//      Channel: %0s, EMPTY", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~AESL_inst_kernel0.B_V_c_U.if_full_n) begin
                                $display("//      Channel: %0s, FULL", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                            else begin
                                $display("//      Channel: %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status NULL");
                            end
                        end
                        if (((AESL_inst_kernel0.kernel0_entry6_U0_ap_ready_count[0]) & AESL_inst_kernel0.kernel0_entry6_U0.ap_idle & ~(AESL_inst_kernel0.B_IO_L3_in_U0_ap_ready_count[0]))) begin
                            chan_path = "";
                            if (((AESL_inst_kernel0.kernel0_entry6_U0_ap_ready_count[0]) & AESL_inst_kernel0.kernel0_entry6_U0.ap_idle & ~(AESL_inst_kernel0.B_IO_L3_in_U0_ap_ready_count[0]))) begin
                                $display("//      Deadlocked by sync logic between input processes");
                                $display("//      Please increase channel depth");
                            end
                        end
                    end
                    21: begin
                        if (~AESL_inst_kernel0.kernel0_entry6_U0.C_V_out_blk_n) begin
                            chan_path = "kernel0.C_V_c_U";
                            if (~AESL_inst_kernel0.C_V_c_U.if_empty_n) begin
                                $display("//      Channel: %0s, EMPTY", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~AESL_inst_kernel0.C_V_c_U.if_full_n) begin
                                $display("//      Channel: %0s, FULL", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                            else begin
                                $display("//      Channel: %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status NULL");
                            end
                        end
                        if ((~AESL_inst_kernel0.start_for_C_drainrcU_U.if_full_n & AESL_inst_kernel0.C_drain_IO_L3_out_U0.ap_done)) begin
                            chan_path = "";
                            if ((~AESL_inst_kernel0.start_for_C_drainrcU_U.if_full_n & AESL_inst_kernel0.C_drain_IO_L3_out_U0.ap_done)) begin
                                $display("//      Deadlock detected: can be a false alarm due to leftover data,");
                                $display("//      please try cosim_design -disable_deadlock_detection");
                            end
                        end
                    end
                    endcase
                end
                1 : begin
                    case(index2)
                    0: begin
                        if (~AESL_inst_kernel0.A_IO_L3_in_U0.A_V_offset_blk_n) begin
                            chan_path = "kernel0.A_V_c_U";
                            if (~AESL_inst_kernel0.A_V_c_U.if_empty_n) begin
                                $display("//      Channel: %0s, EMPTY", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~AESL_inst_kernel0.A_V_c_U.if_full_n) begin
                                $display("//      Channel: %0s, FULL", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                            else begin
                                $display("//      Channel: %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status NULL");
                            end
                        end
                        if (((AESL_inst_kernel0.A_IO_L3_in_U0_ap_ready_count[0]) & AESL_inst_kernel0.A_IO_L3_in_U0.ap_idle & ~(AESL_inst_kernel0.kernel0_entry6_U0_ap_ready_count[0]))) begin
                            chan_path = "";
                            if (((AESL_inst_kernel0.A_IO_L3_in_U0_ap_ready_count[0]) & AESL_inst_kernel0.A_IO_L3_in_U0.ap_idle & ~(AESL_inst_kernel0.kernel0_entry6_U0_ap_ready_count[0]))) begin
                                $display("//      Deadlocked by sync logic between input processes");
                                $display("//      Please increase channel depth");
                            end
                        end
                    end
                    2: begin
                        if (~AESL_inst_kernel0.A_IO_L3_in_U0.fifo_A_local_out_V_V_blk_n) begin
                            chan_path = "kernel0.fifo_A_A_IO_L2_in_0_s_U";
                            if (~AESL_inst_kernel0.fifo_A_A_IO_L2_in_0_s_U.if_empty_n) begin
                                $display("//      Channel: %0s, EMPTY", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~AESL_inst_kernel0.fifo_A_A_IO_L2_in_0_s_U.if_full_n) begin
                                $display("//      Channel: %0s, FULL", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                            else begin
                                $display("//      Channel: %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status NULL");
                            end
                        end
                        if ((~AESL_inst_kernel0.start_for_A_IO_L2sc4_U.if_full_n & AESL_inst_kernel0.A_IO_L2_in_U0.ap_done)) begin
                            chan_path = "";
                            if ((~AESL_inst_kernel0.start_for_A_IO_L2sc4_U.if_full_n & AESL_inst_kernel0.A_IO_L2_in_U0.ap_done)) begin
                                $display("//      Deadlock detected: can be a false alarm due to leftover data,");
                                $display("//      please try cosim_design -disable_deadlock_detection");
                            end
                        end
                    end
                    4: begin
                        if (((AESL_inst_kernel0.A_IO_L3_in_U0_ap_ready_count[0]) & AESL_inst_kernel0.A_IO_L3_in_U0.ap_idle & ~(AESL_inst_kernel0.B_IO_L3_in_U0_ap_ready_count[0]))) begin
                            chan_path = "";
                            if (((AESL_inst_kernel0.A_IO_L3_in_U0_ap_ready_count[0]) & AESL_inst_kernel0.A_IO_L3_in_U0.ap_idle & ~(AESL_inst_kernel0.B_IO_L3_in_U0_ap_ready_count[0]))) begin
                                $display("//      Deadlocked by sync logic between input processes");
                                $display("//      Please increase channel depth");
                            end
                        end
                    end
                    endcase
                end
                2 : begin
                    case(index2)
                    1: begin
                        if (~AESL_inst_kernel0.A_IO_L2_in_U0.grp_A_IO_L2_in_inter_tra_1_fu_139.fifo_A_in_V_V_blk_n) begin
                            chan_path = "kernel0.fifo_A_A_IO_L2_in_0_s_U";
                            if (~AESL_inst_kernel0.fifo_A_A_IO_L2_in_0_s_U.if_empty_n) begin
                                $display("//      Channel: %0s, EMPTY", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~AESL_inst_kernel0.fifo_A_A_IO_L2_in_0_s_U.if_full_n) begin
                                $display("//      Channel: %0s, FULL", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                            else begin
                                $display("//      Channel: %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status NULL");
                            end
                        end
                        if ((~AESL_inst_kernel0.start_for_A_IO_L2sc4_U.if_empty_n & (AESL_inst_kernel0.A_IO_L2_in_U0.ap_ready | AESL_inst_kernel0.A_IO_L2_in_U0.ap_idle))) begin
                            chan_path = "";
                            if ((~AESL_inst_kernel0.start_for_A_IO_L2sc4_U.if_empty_n & (AESL_inst_kernel0.A_IO_L2_in_U0.ap_ready | AESL_inst_kernel0.A_IO_L2_in_U0.ap_idle))) begin
                                $display("//      Deadlock detected: can be a false alarm due to leftover data,");
                                $display("//      please try cosim_design -disable_deadlock_detection");
                            end
                        end
                    end
                    3: begin
                        if (~AESL_inst_kernel0.A_IO_L2_in_U0.grp_A_IO_L2_in_inter_tra_1_fu_139.fifo_A_out_V_V_blk_n) begin
                            chan_path = "kernel0.fifo_A_A_IO_L2_in_1_s_U";
                            if (~AESL_inst_kernel0.fifo_A_A_IO_L2_in_1_s_U.if_empty_n) begin
                                $display("//      Channel: %0s, EMPTY", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~AESL_inst_kernel0.fifo_A_A_IO_L2_in_1_s_U.if_full_n) begin
                                $display("//      Channel: %0s, FULL", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                            else begin
                                $display("//      Channel: %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status NULL");
                            end
                        end
                        if ((~AESL_inst_kernel0.start_for_A_IO_L2tde_U.if_full_n & AESL_inst_kernel0.A_IO_L2_in_tail_U0.ap_done)) begin
                            chan_path = "";
                            if ((~AESL_inst_kernel0.start_for_A_IO_L2tde_U.if_full_n & AESL_inst_kernel0.A_IO_L2_in_tail_U0.ap_done)) begin
                                $display("//      Deadlock detected: can be a false alarm due to leftover data,");
                                $display("//      please try cosim_design -disable_deadlock_detection");
                            end
                        end
                    end
                    7: begin
                        if (~AESL_inst_kernel0.A_IO_L2_in_U0.grp_A_IO_L2_in_intra_tra_fu_131.fifo_A_local_out_V_V_blk_n) begin
                            chan_path = "kernel0.fifo_A_PE_0_0_V_V_U";
                            if (~AESL_inst_kernel0.fifo_A_PE_0_0_V_V_U.if_empty_n) begin
                                $display("//      Channel: %0s, EMPTY", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~AESL_inst_kernel0.fifo_A_PE_0_0_V_V_U.if_full_n) begin
                                $display("//      Channel: %0s, FULL", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                            else begin
                                $display("//      Channel: %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status NULL");
                            end
                        end
                        if ((~AESL_inst_kernel0.start_for_PE139_U0_U.if_full_n & AESL_inst_kernel0.PE139_U0.ap_done)) begin
                            chan_path = "";
                            if ((~AESL_inst_kernel0.start_for_PE139_U0_U.if_full_n & AESL_inst_kernel0.PE139_U0.ap_done)) begin
                                $display("//      Deadlock detected: can be a false alarm due to leftover data,");
                                $display("//      please try cosim_design -disable_deadlock_detection");
                            end
                        end
                    end
                    endcase
                end
                3 : begin
                    case(index2)
                    2: begin
                        if (~AESL_inst_kernel0.A_IO_L2_in_tail_U0.grp_A_IO_L2_in_inter_tra_fu_125.fifo_A_in_V_V_blk_n) begin
                            chan_path = "kernel0.fifo_A_A_IO_L2_in_1_s_U";
                            if (~AESL_inst_kernel0.fifo_A_A_IO_L2_in_1_s_U.if_empty_n) begin
                                $display("//      Channel: %0s, EMPTY", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~AESL_inst_kernel0.fifo_A_A_IO_L2_in_1_s_U.if_full_n) begin
                                $display("//      Channel: %0s, FULL", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                            else begin
                                $display("//      Channel: %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status NULL");
                            end
                        end
                        if ((~AESL_inst_kernel0.start_for_A_IO_L2tde_U.if_empty_n & (AESL_inst_kernel0.A_IO_L2_in_tail_U0.ap_ready | AESL_inst_kernel0.A_IO_L2_in_tail_U0.ap_idle))) begin
                            chan_path = "";
                            if ((~AESL_inst_kernel0.start_for_A_IO_L2tde_U.if_empty_n & (AESL_inst_kernel0.A_IO_L2_in_tail_U0.ap_ready | AESL_inst_kernel0.A_IO_L2_in_tail_U0.ap_idle))) begin
                                $display("//      Deadlock detected: can be a false alarm due to leftover data,");
                                $display("//      please try cosim_design -disable_deadlock_detection");
                            end
                        end
                    end
                    10: begin
                        if (~AESL_inst_kernel0.A_IO_L2_in_tail_U0.grp_A_IO_L2_in_intra_tra_fu_117.fifo_A_local_out_V_V_blk_n) begin
                            chan_path = "kernel0.fifo_A_PE_1_0_V_V_U";
                            if (~AESL_inst_kernel0.fifo_A_PE_1_0_V_V_U.if_empty_n) begin
                                $display("//      Channel: %0s, EMPTY", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~AESL_inst_kernel0.fifo_A_PE_1_0_V_V_U.if_full_n) begin
                                $display("//      Channel: %0s, FULL", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                            else begin
                                $display("//      Channel: %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status NULL");
                            end
                        end
                        if ((~AESL_inst_kernel0.start_for_PE142_U0_U.if_full_n & AESL_inst_kernel0.PE142_U0.ap_done)) begin
                            chan_path = "";
                            if ((~AESL_inst_kernel0.start_for_PE142_U0_U.if_full_n & AESL_inst_kernel0.PE142_U0.ap_done)) begin
                                $display("//      Deadlock detected: can be a false alarm due to leftover data,");
                                $display("//      please try cosim_design -disable_deadlock_detection");
                            end
                        end
                    end
                    endcase
                end
                4 : begin
                    case(index2)
                    0: begin
                        if (~AESL_inst_kernel0.B_IO_L3_in_U0.B_V_offset_blk_n) begin
                            chan_path = "kernel0.B_V_c_U";
                            if (~AESL_inst_kernel0.B_V_c_U.if_empty_n) begin
                                $display("//      Channel: %0s, EMPTY", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~AESL_inst_kernel0.B_V_c_U.if_full_n) begin
                                $display("//      Channel: %0s, FULL", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                            else begin
                                $display("//      Channel: %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status NULL");
                            end
                        end
                        if (((AESL_inst_kernel0.B_IO_L3_in_U0_ap_ready_count[0]) & AESL_inst_kernel0.B_IO_L3_in_U0.ap_idle & ~(AESL_inst_kernel0.kernel0_entry6_U0_ap_ready_count[0]))) begin
                            chan_path = "";
                            if (((AESL_inst_kernel0.B_IO_L3_in_U0_ap_ready_count[0]) & AESL_inst_kernel0.B_IO_L3_in_U0.ap_idle & ~(AESL_inst_kernel0.kernel0_entry6_U0_ap_ready_count[0]))) begin
                                $display("//      Deadlocked by sync logic between input processes");
                                $display("//      Please increase channel depth");
                            end
                        end
                    end
                    5: begin
                        if (~AESL_inst_kernel0.B_IO_L3_in_U0.fifo_B_local_out_V_V_blk_n) begin
                            chan_path = "kernel0.fifo_B_B_IO_L2_in_0_s_U";
                            if (~AESL_inst_kernel0.fifo_B_B_IO_L2_in_0_s_U.if_empty_n) begin
                                $display("//      Channel: %0s, EMPTY", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~AESL_inst_kernel0.fifo_B_B_IO_L2_in_0_s_U.if_full_n) begin
                                $display("//      Channel: %0s, FULL", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                            else begin
                                $display("//      Channel: %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status NULL");
                            end
                        end
                        if ((~AESL_inst_kernel0.start_for_B_IO_L2udo_U.if_full_n & AESL_inst_kernel0.B_IO_L2_in_U0.ap_done)) begin
                            chan_path = "";
                            if ((~AESL_inst_kernel0.start_for_B_IO_L2udo_U.if_full_n & AESL_inst_kernel0.B_IO_L2_in_U0.ap_done)) begin
                                $display("//      Deadlock detected: can be a false alarm due to leftover data,");
                                $display("//      please try cosim_design -disable_deadlock_detection");
                            end
                        end
                    end
                    1: begin
                        if (((AESL_inst_kernel0.B_IO_L3_in_U0_ap_ready_count[0]) & AESL_inst_kernel0.B_IO_L3_in_U0.ap_idle & ~(AESL_inst_kernel0.A_IO_L3_in_U0_ap_ready_count[0]))) begin
                            chan_path = "";
                            if (((AESL_inst_kernel0.B_IO_L3_in_U0_ap_ready_count[0]) & AESL_inst_kernel0.B_IO_L3_in_U0.ap_idle & ~(AESL_inst_kernel0.A_IO_L3_in_U0_ap_ready_count[0]))) begin
                                $display("//      Deadlocked by sync logic between input processes");
                                $display("//      Please increase channel depth");
                            end
                        end
                    end
                    endcase
                end
                5 : begin
                    case(index2)
                    4: begin
                        if (~AESL_inst_kernel0.B_IO_L2_in_U0.grp_B_IO_L2_in_inter_tra_1_fu_139.fifo_B_in_V_V_blk_n) begin
                            chan_path = "kernel0.fifo_B_B_IO_L2_in_0_s_U";
                            if (~AESL_inst_kernel0.fifo_B_B_IO_L2_in_0_s_U.if_empty_n) begin
                                $display("//      Channel: %0s, EMPTY", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~AESL_inst_kernel0.fifo_B_B_IO_L2_in_0_s_U.if_full_n) begin
                                $display("//      Channel: %0s, FULL", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                            else begin
                                $display("//      Channel: %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status NULL");
                            end
                        end
                        if ((~AESL_inst_kernel0.start_for_B_IO_L2udo_U.if_empty_n & (AESL_inst_kernel0.B_IO_L2_in_U0.ap_ready | AESL_inst_kernel0.B_IO_L2_in_U0.ap_idle))) begin
                            chan_path = "";
                            if ((~AESL_inst_kernel0.start_for_B_IO_L2udo_U.if_empty_n & (AESL_inst_kernel0.B_IO_L2_in_U0.ap_ready | AESL_inst_kernel0.B_IO_L2_in_U0.ap_idle))) begin
                                $display("//      Deadlock detected: can be a false alarm due to leftover data,");
                                $display("//      please try cosim_design -disable_deadlock_detection");
                            end
                        end
                    end
                    6: begin
                        if (~AESL_inst_kernel0.B_IO_L2_in_U0.grp_B_IO_L2_in_inter_tra_1_fu_139.fifo_B_out_V_V_blk_n) begin
                            chan_path = "kernel0.fifo_B_B_IO_L2_in_1_s_U";
                            if (~AESL_inst_kernel0.fifo_B_B_IO_L2_in_1_s_U.if_empty_n) begin
                                $display("//      Channel: %0s, EMPTY", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~AESL_inst_kernel0.fifo_B_B_IO_L2_in_1_s_U.if_full_n) begin
                                $display("//      Channel: %0s, FULL", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                            else begin
                                $display("//      Channel: %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status NULL");
                            end
                        end
                        if ((~AESL_inst_kernel0.start_for_B_IO_L2vdy_U.if_full_n & AESL_inst_kernel0.B_IO_L2_in_tail_U0.ap_done)) begin
                            chan_path = "";
                            if ((~AESL_inst_kernel0.start_for_B_IO_L2vdy_U.if_full_n & AESL_inst_kernel0.B_IO_L2_in_tail_U0.ap_done)) begin
                                $display("//      Deadlock detected: can be a false alarm due to leftover data,");
                                $display("//      please try cosim_design -disable_deadlock_detection");
                            end
                        end
                    end
                    7: begin
                        if (~AESL_inst_kernel0.B_IO_L2_in_U0.grp_B_IO_L2_in_intra_tra_fu_131.fifo_B_local_out_V_V_blk_n) begin
                            chan_path = "kernel0.fifo_B_PE_0_0_V_V_U";
                            if (~AESL_inst_kernel0.fifo_B_PE_0_0_V_V_U.if_empty_n) begin
                                $display("//      Channel: %0s, EMPTY", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~AESL_inst_kernel0.fifo_B_PE_0_0_V_V_U.if_full_n) begin
                                $display("//      Channel: %0s, FULL", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                            else begin
                                $display("//      Channel: %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status NULL");
                            end
                        end
                    end
                    endcase
                end
                6 : begin
                    case(index2)
                    5: begin
                        if (~AESL_inst_kernel0.B_IO_L2_in_tail_U0.grp_B_IO_L2_in_inter_tra_fu_125.fifo_B_in_V_V_blk_n) begin
                            chan_path = "kernel0.fifo_B_B_IO_L2_in_1_s_U";
                            if (~AESL_inst_kernel0.fifo_B_B_IO_L2_in_1_s_U.if_empty_n) begin
                                $display("//      Channel: %0s, EMPTY", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~AESL_inst_kernel0.fifo_B_B_IO_L2_in_1_s_U.if_full_n) begin
                                $display("//      Channel: %0s, FULL", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                            else begin
                                $display("//      Channel: %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status NULL");
                            end
                        end
                        if ((~AESL_inst_kernel0.start_for_B_IO_L2vdy_U.if_empty_n & (AESL_inst_kernel0.B_IO_L2_in_tail_U0.ap_ready | AESL_inst_kernel0.B_IO_L2_in_tail_U0.ap_idle))) begin
                            chan_path = "";
                            if ((~AESL_inst_kernel0.start_for_B_IO_L2vdy_U.if_empty_n & (AESL_inst_kernel0.B_IO_L2_in_tail_U0.ap_ready | AESL_inst_kernel0.B_IO_L2_in_tail_U0.ap_idle))) begin
                                $display("//      Deadlock detected: can be a false alarm due to leftover data,");
                                $display("//      please try cosim_design -disable_deadlock_detection");
                            end
                        end
                    end
                    8: begin
                        if (~AESL_inst_kernel0.B_IO_L2_in_tail_U0.grp_B_IO_L2_in_intra_tra_fu_117.fifo_B_local_out_V_V_blk_n) begin
                            chan_path = "kernel0.fifo_B_PE_0_1_V_V_U";
                            if (~AESL_inst_kernel0.fifo_B_PE_0_1_V_V_U.if_empty_n) begin
                                $display("//      Channel: %0s, EMPTY", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~AESL_inst_kernel0.fifo_B_PE_0_1_V_V_U.if_full_n) begin
                                $display("//      Channel: %0s, FULL", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                            else begin
                                $display("//      Channel: %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status NULL");
                            end
                        end
                        if ((~AESL_inst_kernel0.start_for_PE140_U0_U.if_full_n & AESL_inst_kernel0.PE140_U0.ap_done)) begin
                            chan_path = "";
                            if ((~AESL_inst_kernel0.start_for_PE140_U0_U.if_full_n & AESL_inst_kernel0.PE140_U0.ap_done)) begin
                                $display("//      Deadlock detected: can be a false alarm due to leftover data,");
                                $display("//      please try cosim_design -disable_deadlock_detection");
                            end
                        end
                    end
                    endcase
                end
                7 : begin
                    case(index2)
                    2: begin
                        if (~AESL_inst_kernel0.PE139_U0.fifo_A_in_V_V_blk_n) begin
                            chan_path = "kernel0.fifo_A_PE_0_0_V_V_U";
                            if (~AESL_inst_kernel0.fifo_A_PE_0_0_V_V_U.if_empty_n) begin
                                $display("//      Channel: %0s, EMPTY", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~AESL_inst_kernel0.fifo_A_PE_0_0_V_V_U.if_full_n) begin
                                $display("//      Channel: %0s, FULL", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                            else begin
                                $display("//      Channel: %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status NULL");
                            end
                        end
                        if ((~AESL_inst_kernel0.start_for_PE139_U0_U.if_empty_n & (AESL_inst_kernel0.PE139_U0.ap_ready | AESL_inst_kernel0.PE139_U0.ap_idle))) begin
                            chan_path = "";
                            if ((~AESL_inst_kernel0.start_for_PE139_U0_U.if_empty_n & (AESL_inst_kernel0.PE139_U0.ap_ready | AESL_inst_kernel0.PE139_U0.ap_idle))) begin
                                $display("//      Deadlock detected: can be a false alarm due to leftover data,");
                                $display("//      please try cosim_design -disable_deadlock_detection");
                            end
                        end
                    end
                    8: begin
                        if (~AESL_inst_kernel0.PE139_U0.fifo_A_out_V_V_blk_n) begin
                            chan_path = "kernel0.fifo_A_PE_0_1_V_V_U";
                            if (~AESL_inst_kernel0.fifo_A_PE_0_1_V_V_U.if_empty_n) begin
                                $display("//      Channel: %0s, EMPTY", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~AESL_inst_kernel0.fifo_A_PE_0_1_V_V_U.if_full_n) begin
                                $display("//      Channel: %0s, FULL", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                            else begin
                                $display("//      Channel: %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status NULL");
                            end
                        end
                    end
                    5: begin
                        if (~AESL_inst_kernel0.PE139_U0.fifo_B_in_V_V_blk_n) begin
                            chan_path = "kernel0.fifo_B_PE_0_0_V_V_U";
                            if (~AESL_inst_kernel0.fifo_B_PE_0_0_V_V_U.if_empty_n) begin
                                $display("//      Channel: %0s, EMPTY", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~AESL_inst_kernel0.fifo_B_PE_0_0_V_V_U.if_full_n) begin
                                $display("//      Channel: %0s, FULL", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                            else begin
                                $display("//      Channel: %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status NULL");
                            end
                        end
                    end
                    10: begin
                        if (~AESL_inst_kernel0.PE139_U0.fifo_B_out_V_V_blk_n) begin
                            chan_path = "kernel0.fifo_B_PE_1_0_V_V_U";
                            if (~AESL_inst_kernel0.fifo_B_PE_1_0_V_V_U.if_empty_n) begin
                                $display("//      Channel: %0s, EMPTY", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~AESL_inst_kernel0.fifo_B_PE_1_0_V_V_U.if_full_n) begin
                                $display("//      Channel: %0s, FULL", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                            else begin
                                $display("//      Channel: %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status NULL");
                            end
                        end
                    end
                    15: begin
                        if (~AESL_inst_kernel0.PE139_U0.fifo_C_drain_out_V_blk_n) begin
                            chan_path = "kernel0.fifo_C_drain_PE_0_0_s_U";
                            if (~AESL_inst_kernel0.fifo_C_drain_PE_0_0_s_U.if_empty_n) begin
                                $display("//      Channel: %0s, EMPTY", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~AESL_inst_kernel0.fifo_C_drain_PE_0_0_s_U.if_full_n) begin
                                $display("//      Channel: %0s, FULL", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                            else begin
                                $display("//      Channel: %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status NULL");
                            end
                        end
                        if ((~AESL_inst_kernel0.start_for_C_drainwdI_U.if_full_n & AESL_inst_kernel0.C_drain_IO_L1_out_he_U0.ap_done)) begin
                            chan_path = "";
                            if ((~AESL_inst_kernel0.start_for_C_drainwdI_U.if_full_n & AESL_inst_kernel0.C_drain_IO_L1_out_he_U0.ap_done)) begin
                                $display("//      Deadlock detected: can be a false alarm due to leftover data,");
                                $display("//      please try cosim_design -disable_deadlock_detection");
                            end
                        end
                    end
                    endcase
                end
                8 : begin
                    case(index2)
                    7: begin
                        if (~AESL_inst_kernel0.PE140_U0.fifo_A_in_V_V_blk_n) begin
                            chan_path = "kernel0.fifo_A_PE_0_1_V_V_U";
                            if (~AESL_inst_kernel0.fifo_A_PE_0_1_V_V_U.if_empty_n) begin
                                $display("//      Channel: %0s, EMPTY", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~AESL_inst_kernel0.fifo_A_PE_0_1_V_V_U.if_full_n) begin
                                $display("//      Channel: %0s, FULL", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                            else begin
                                $display("//      Channel: %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status NULL");
                            end
                        end
                    end
                    9: begin
                        if (~AESL_inst_kernel0.PE140_U0.fifo_A_out_V_V_blk_n) begin
                            chan_path = "kernel0.fifo_A_PE_0_2_V_V_U";
                            if (~AESL_inst_kernel0.fifo_A_PE_0_2_V_V_U.if_empty_n) begin
                                $display("//      Channel: %0s, EMPTY", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~AESL_inst_kernel0.fifo_A_PE_0_2_V_V_U.if_full_n) begin
                                $display("//      Channel: %0s, FULL", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                            else begin
                                $display("//      Channel: %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status NULL");
                            end
                        end
                        if ((~AESL_inst_kernel0.start_for_PE_A_duxdS_U.if_full_n & AESL_inst_kernel0.PE_A_dummy141_U0.ap_done)) begin
                            chan_path = "";
                            if ((~AESL_inst_kernel0.start_for_PE_A_duxdS_U.if_full_n & AESL_inst_kernel0.PE_A_dummy141_U0.ap_done)) begin
                                $display("//      Deadlock detected: can be a false alarm due to leftover data,");
                                $display("//      please try cosim_design -disable_deadlock_detection");
                            end
                        end
                    end
                    6: begin
                        if (~AESL_inst_kernel0.PE140_U0.fifo_B_in_V_V_blk_n) begin
                            chan_path = "kernel0.fifo_B_PE_0_1_V_V_U";
                            if (~AESL_inst_kernel0.fifo_B_PE_0_1_V_V_U.if_empty_n) begin
                                $display("//      Channel: %0s, EMPTY", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~AESL_inst_kernel0.fifo_B_PE_0_1_V_V_U.if_full_n) begin
                                $display("//      Channel: %0s, FULL", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                            else begin
                                $display("//      Channel: %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status NULL");
                            end
                        end
                        if ((~AESL_inst_kernel0.start_for_PE140_U0_U.if_empty_n & (AESL_inst_kernel0.PE140_U0.ap_ready | AESL_inst_kernel0.PE140_U0.ap_idle))) begin
                            chan_path = "";
                            if ((~AESL_inst_kernel0.start_for_PE140_U0_U.if_empty_n & (AESL_inst_kernel0.PE140_U0.ap_ready | AESL_inst_kernel0.PE140_U0.ap_idle))) begin
                                $display("//      Deadlock detected: can be a false alarm due to leftover data,");
                                $display("//      please try cosim_design -disable_deadlock_detection");
                            end
                        end
                    end
                    12: begin
                        if (~AESL_inst_kernel0.PE140_U0.fifo_B_out_V_V_blk_n) begin
                            chan_path = "kernel0.fifo_B_PE_1_1_V_V_U";
                            if (~AESL_inst_kernel0.fifo_B_PE_1_1_V_V_U.if_empty_n) begin
                                $display("//      Channel: %0s, EMPTY", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~AESL_inst_kernel0.fifo_B_PE_1_1_V_V_U.if_full_n) begin
                                $display("//      Channel: %0s, FULL", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                            else begin
                                $display("//      Channel: %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status NULL");
                            end
                        end
                        if ((~AESL_inst_kernel0.start_for_PE_U0_U.if_full_n & AESL_inst_kernel0.PE_U0.ap_done)) begin
                            chan_path = "";
                            if ((~AESL_inst_kernel0.start_for_PE_U0_U.if_full_n & AESL_inst_kernel0.PE_U0.ap_done)) begin
                                $display("//      Deadlock detected: can be a false alarm due to leftover data,");
                                $display("//      please try cosim_design -disable_deadlock_detection");
                            end
                        end
                    end
                    17: begin
                        if (~AESL_inst_kernel0.PE140_U0.fifo_C_drain_out_V_blk_n) begin
                            chan_path = "kernel0.fifo_C_drain_PE_0_1_s_U";
                            if (~AESL_inst_kernel0.fifo_C_drain_PE_0_1_s_U.if_empty_n) begin
                                $display("//      Channel: %0s, EMPTY", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~AESL_inst_kernel0.fifo_C_drain_PE_0_1_s_U.if_full_n) begin
                                $display("//      Channel: %0s, FULL", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                            else begin
                                $display("//      Channel: %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status NULL");
                            end
                        end
                        if ((~AESL_inst_kernel0.start_for_C_drainyd2_U.if_full_n & AESL_inst_kernel0.C_drain_IO_L1_out_he_1_U0.ap_done)) begin
                            chan_path = "";
                            if ((~AESL_inst_kernel0.start_for_C_drainyd2_U.if_full_n & AESL_inst_kernel0.C_drain_IO_L1_out_he_1_U0.ap_done)) begin
                                $display("//      Deadlock detected: can be a false alarm due to leftover data,");
                                $display("//      please try cosim_design -disable_deadlock_detection");
                            end
                        end
                    end
                    endcase
                end
                9 : begin
                    case(index2)
                    8: begin
                        if (~AESL_inst_kernel0.PE_A_dummy141_U0.fifo_A_in_V_V_blk_n) begin
                            chan_path = "kernel0.fifo_A_PE_0_2_V_V_U";
                            if (~AESL_inst_kernel0.fifo_A_PE_0_2_V_V_U.if_empty_n) begin
                                $display("//      Channel: %0s, EMPTY", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~AESL_inst_kernel0.fifo_A_PE_0_2_V_V_U.if_full_n) begin
                                $display("//      Channel: %0s, FULL", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                            else begin
                                $display("//      Channel: %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status NULL");
                            end
                        end
                        if ((~AESL_inst_kernel0.start_for_PE_A_duxdS_U.if_empty_n & (AESL_inst_kernel0.PE_A_dummy141_U0.ap_ready | AESL_inst_kernel0.PE_A_dummy141_U0.ap_idle))) begin
                            chan_path = "";
                            if ((~AESL_inst_kernel0.start_for_PE_A_duxdS_U.if_empty_n & (AESL_inst_kernel0.PE_A_dummy141_U0.ap_ready | AESL_inst_kernel0.PE_A_dummy141_U0.ap_idle))) begin
                                $display("//      Deadlock detected: can be a false alarm due to leftover data,");
                                $display("//      please try cosim_design -disable_deadlock_detection");
                            end
                        end
                    end
                    endcase
                end
                10 : begin
                    case(index2)
                    3: begin
                        if (~AESL_inst_kernel0.PE142_U0.fifo_A_in_V_V_blk_n) begin
                            chan_path = "kernel0.fifo_A_PE_1_0_V_V_U";
                            if (~AESL_inst_kernel0.fifo_A_PE_1_0_V_V_U.if_empty_n) begin
                                $display("//      Channel: %0s, EMPTY", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~AESL_inst_kernel0.fifo_A_PE_1_0_V_V_U.if_full_n) begin
                                $display("//      Channel: %0s, FULL", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                            else begin
                                $display("//      Channel: %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status NULL");
                            end
                        end
                        if ((~AESL_inst_kernel0.start_for_PE142_U0_U.if_empty_n & (AESL_inst_kernel0.PE142_U0.ap_ready | AESL_inst_kernel0.PE142_U0.ap_idle))) begin
                            chan_path = "";
                            if ((~AESL_inst_kernel0.start_for_PE142_U0_U.if_empty_n & (AESL_inst_kernel0.PE142_U0.ap_ready | AESL_inst_kernel0.PE142_U0.ap_idle))) begin
                                $display("//      Deadlock detected: can be a false alarm due to leftover data,");
                                $display("//      please try cosim_design -disable_deadlock_detection");
                            end
                        end
                    end
                    12: begin
                        if (~AESL_inst_kernel0.PE142_U0.fifo_A_out_V_V_blk_n) begin
                            chan_path = "kernel0.fifo_A_PE_1_1_V_V_U";
                            if (~AESL_inst_kernel0.fifo_A_PE_1_1_V_V_U.if_empty_n) begin
                                $display("//      Channel: %0s, EMPTY", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~AESL_inst_kernel0.fifo_A_PE_1_1_V_V_U.if_full_n) begin
                                $display("//      Channel: %0s, FULL", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                            else begin
                                $display("//      Channel: %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status NULL");
                            end
                        end
                    end
                    7: begin
                        if (~AESL_inst_kernel0.PE142_U0.fifo_B_in_V_V_blk_n) begin
                            chan_path = "kernel0.fifo_B_PE_1_0_V_V_U";
                            if (~AESL_inst_kernel0.fifo_B_PE_1_0_V_V_U.if_empty_n) begin
                                $display("//      Channel: %0s, EMPTY", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~AESL_inst_kernel0.fifo_B_PE_1_0_V_V_U.if_full_n) begin
                                $display("//      Channel: %0s, FULL", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                            else begin
                                $display("//      Channel: %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status NULL");
                            end
                        end
                    end
                    11: begin
                        if (~AESL_inst_kernel0.PE142_U0.fifo_B_out_V_V_blk_n) begin
                            chan_path = "kernel0.fifo_B_PE_2_0_V_V_U";
                            if (~AESL_inst_kernel0.fifo_B_PE_2_0_V_V_U.if_empty_n) begin
                                $display("//      Channel: %0s, EMPTY", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~AESL_inst_kernel0.fifo_B_PE_2_0_V_V_U.if_full_n) begin
                                $display("//      Channel: %0s, FULL", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                            else begin
                                $display("//      Channel: %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status NULL");
                            end
                        end
                        if ((~AESL_inst_kernel0.start_for_PE_B_duzec_U.if_full_n & AESL_inst_kernel0.PE_B_dummy143_U0.ap_done)) begin
                            chan_path = "";
                            if ((~AESL_inst_kernel0.start_for_PE_B_duzec_U.if_full_n & AESL_inst_kernel0.PE_B_dummy143_U0.ap_done)) begin
                                $display("//      Deadlock detected: can be a false alarm due to leftover data,");
                                $display("//      please try cosim_design -disable_deadlock_detection");
                            end
                        end
                    end
                    16: begin
                        if (~AESL_inst_kernel0.PE142_U0.fifo_C_drain_out_V_blk_n) begin
                            chan_path = "kernel0.fifo_C_drain_PE_1_0_s_U";
                            if (~AESL_inst_kernel0.fifo_C_drain_PE_1_0_s_U.if_empty_n) begin
                                $display("//      Channel: %0s, EMPTY", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~AESL_inst_kernel0.fifo_C_drain_PE_1_0_s_U.if_full_n) begin
                                $display("//      Channel: %0s, FULL", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                            else begin
                                $display("//      Channel: %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status NULL");
                            end
                        end
                        if ((~AESL_inst_kernel0.start_for_C_drainAem_U.if_full_n & AESL_inst_kernel0.C_drain_IO_L1_out145_U0.ap_done)) begin
                            chan_path = "";
                            if ((~AESL_inst_kernel0.start_for_C_drainAem_U.if_full_n & AESL_inst_kernel0.C_drain_IO_L1_out145_U0.ap_done)) begin
                                $display("//      Deadlock detected: can be a false alarm due to leftover data,");
                                $display("//      please try cosim_design -disable_deadlock_detection");
                            end
                        end
                    end
                    endcase
                end
                11 : begin
                    case(index2)
                    10: begin
                        if (~AESL_inst_kernel0.PE_B_dummy143_U0.fifo_B_in_V_V_blk_n) begin
                            chan_path = "kernel0.fifo_B_PE_2_0_V_V_U";
                            if (~AESL_inst_kernel0.fifo_B_PE_2_0_V_V_U.if_empty_n) begin
                                $display("//      Channel: %0s, EMPTY", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~AESL_inst_kernel0.fifo_B_PE_2_0_V_V_U.if_full_n) begin
                                $display("//      Channel: %0s, FULL", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                            else begin
                                $display("//      Channel: %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status NULL");
                            end
                        end
                        if ((~AESL_inst_kernel0.start_for_PE_B_duzec_U.if_empty_n & (AESL_inst_kernel0.PE_B_dummy143_U0.ap_ready | AESL_inst_kernel0.PE_B_dummy143_U0.ap_idle))) begin
                            chan_path = "";
                            if ((~AESL_inst_kernel0.start_for_PE_B_duzec_U.if_empty_n & (AESL_inst_kernel0.PE_B_dummy143_U0.ap_ready | AESL_inst_kernel0.PE_B_dummy143_U0.ap_idle))) begin
                                $display("//      Deadlock detected: can be a false alarm due to leftover data,");
                                $display("//      please try cosim_design -disable_deadlock_detection");
                            end
                        end
                    end
                    endcase
                end
                12 : begin
                    case(index2)
                    10: begin
                        if (~AESL_inst_kernel0.PE_U0.fifo_A_in_V_V_blk_n) begin
                            chan_path = "kernel0.fifo_A_PE_1_1_V_V_U";
                            if (~AESL_inst_kernel0.fifo_A_PE_1_1_V_V_U.if_empty_n) begin
                                $display("//      Channel: %0s, EMPTY", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~AESL_inst_kernel0.fifo_A_PE_1_1_V_V_U.if_full_n) begin
                                $display("//      Channel: %0s, FULL", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                            else begin
                                $display("//      Channel: %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status NULL");
                            end
                        end
                    end
                    13: begin
                        if (~AESL_inst_kernel0.PE_U0.fifo_A_out_V_V_blk_n) begin
                            chan_path = "kernel0.fifo_A_PE_1_2_V_V_U";
                            if (~AESL_inst_kernel0.fifo_A_PE_1_2_V_V_U.if_empty_n) begin
                                $display("//      Channel: %0s, EMPTY", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~AESL_inst_kernel0.fifo_A_PE_1_2_V_V_U.if_full_n) begin
                                $display("//      Channel: %0s, FULL", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                            else begin
                                $display("//      Channel: %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status NULL");
                            end
                        end
                        if ((~AESL_inst_kernel0.start_for_PE_A_duBew_U.if_full_n & AESL_inst_kernel0.PE_A_dummy_U0.ap_done)) begin
                            chan_path = "";
                            if ((~AESL_inst_kernel0.start_for_PE_A_duBew_U.if_full_n & AESL_inst_kernel0.PE_A_dummy_U0.ap_done)) begin
                                $display("//      Deadlock detected: can be a false alarm due to leftover data,");
                                $display("//      please try cosim_design -disable_deadlock_detection");
                            end
                        end
                    end
                    8: begin
                        if (~AESL_inst_kernel0.PE_U0.fifo_B_in_V_V_blk_n) begin
                            chan_path = "kernel0.fifo_B_PE_1_1_V_V_U";
                            if (~AESL_inst_kernel0.fifo_B_PE_1_1_V_V_U.if_empty_n) begin
                                $display("//      Channel: %0s, EMPTY", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~AESL_inst_kernel0.fifo_B_PE_1_1_V_V_U.if_full_n) begin
                                $display("//      Channel: %0s, FULL", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                            else begin
                                $display("//      Channel: %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status NULL");
                            end
                        end
                        if ((~AESL_inst_kernel0.start_for_PE_U0_U.if_empty_n & (AESL_inst_kernel0.PE_U0.ap_ready | AESL_inst_kernel0.PE_U0.ap_idle))) begin
                            chan_path = "";
                            if ((~AESL_inst_kernel0.start_for_PE_U0_U.if_empty_n & (AESL_inst_kernel0.PE_U0.ap_ready | AESL_inst_kernel0.PE_U0.ap_idle))) begin
                                $display("//      Deadlock detected: can be a false alarm due to leftover data,");
                                $display("//      please try cosim_design -disable_deadlock_detection");
                            end
                        end
                    end
                    14: begin
                        if (~AESL_inst_kernel0.PE_U0.fifo_B_out_V_V_blk_n) begin
                            chan_path = "kernel0.fifo_B_PE_2_1_V_V_U";
                            if (~AESL_inst_kernel0.fifo_B_PE_2_1_V_V_U.if_empty_n) begin
                                $display("//      Channel: %0s, EMPTY", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~AESL_inst_kernel0.fifo_B_PE_2_1_V_V_U.if_full_n) begin
                                $display("//      Channel: %0s, FULL", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                            else begin
                                $display("//      Channel: %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status NULL");
                            end
                        end
                        if ((~AESL_inst_kernel0.start_for_PE_B_duCeG_U.if_full_n & AESL_inst_kernel0.PE_B_dummy_U0.ap_done)) begin
                            chan_path = "";
                            if ((~AESL_inst_kernel0.start_for_PE_B_duCeG_U.if_full_n & AESL_inst_kernel0.PE_B_dummy_U0.ap_done)) begin
                                $display("//      Deadlock detected: can be a false alarm due to leftover data,");
                                $display("//      please try cosim_design -disable_deadlock_detection");
                            end
                        end
                    end
                    18: begin
                        if (~AESL_inst_kernel0.PE_U0.fifo_C_drain_out_V_blk_n) begin
                            chan_path = "kernel0.fifo_C_drain_PE_1_1_s_U";
                            if (~AESL_inst_kernel0.fifo_C_drain_PE_1_1_s_U.if_empty_n) begin
                                $display("//      Channel: %0s, EMPTY", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~AESL_inst_kernel0.fifo_C_drain_PE_1_1_s_U.if_full_n) begin
                                $display("//      Channel: %0s, FULL", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                            else begin
                                $display("//      Channel: %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status NULL");
                            end
                        end
                        if ((~AESL_inst_kernel0.start_for_C_drainDeQ_U.if_full_n & AESL_inst_kernel0.C_drain_IO_L1_out_U0.ap_done)) begin
                            chan_path = "";
                            if ((~AESL_inst_kernel0.start_for_C_drainDeQ_U.if_full_n & AESL_inst_kernel0.C_drain_IO_L1_out_U0.ap_done)) begin
                                $display("//      Deadlock detected: can be a false alarm due to leftover data,");
                                $display("//      please try cosim_design -disable_deadlock_detection");
                            end
                        end
                    end
                    endcase
                end
                13 : begin
                    case(index2)
                    12: begin
                        if (~AESL_inst_kernel0.PE_A_dummy_U0.fifo_A_in_V_V_blk_n) begin
                            chan_path = "kernel0.fifo_A_PE_1_2_V_V_U";
                            if (~AESL_inst_kernel0.fifo_A_PE_1_2_V_V_U.if_empty_n) begin
                                $display("//      Channel: %0s, EMPTY", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~AESL_inst_kernel0.fifo_A_PE_1_2_V_V_U.if_full_n) begin
                                $display("//      Channel: %0s, FULL", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                            else begin
                                $display("//      Channel: %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status NULL");
                            end
                        end
                        if ((~AESL_inst_kernel0.start_for_PE_A_duBew_U.if_empty_n & (AESL_inst_kernel0.PE_A_dummy_U0.ap_ready | AESL_inst_kernel0.PE_A_dummy_U0.ap_idle))) begin
                            chan_path = "";
                            if ((~AESL_inst_kernel0.start_for_PE_A_duBew_U.if_empty_n & (AESL_inst_kernel0.PE_A_dummy_U0.ap_ready | AESL_inst_kernel0.PE_A_dummy_U0.ap_idle))) begin
                                $display("//      Deadlock detected: can be a false alarm due to leftover data,");
                                $display("//      please try cosim_design -disable_deadlock_detection");
                            end
                        end
                    end
                    endcase
                end
                14 : begin
                    case(index2)
                    12: begin
                        if (~AESL_inst_kernel0.PE_B_dummy_U0.fifo_B_in_V_V_blk_n) begin
                            chan_path = "kernel0.fifo_B_PE_2_1_V_V_U";
                            if (~AESL_inst_kernel0.fifo_B_PE_2_1_V_V_U.if_empty_n) begin
                                $display("//      Channel: %0s, EMPTY", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~AESL_inst_kernel0.fifo_B_PE_2_1_V_V_U.if_full_n) begin
                                $display("//      Channel: %0s, FULL", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                            else begin
                                $display("//      Channel: %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status NULL");
                            end
                        end
                        if ((~AESL_inst_kernel0.start_for_PE_B_duCeG_U.if_empty_n & (AESL_inst_kernel0.PE_B_dummy_U0.ap_ready | AESL_inst_kernel0.PE_B_dummy_U0.ap_idle))) begin
                            chan_path = "";
                            if ((~AESL_inst_kernel0.start_for_PE_B_duCeG_U.if_empty_n & (AESL_inst_kernel0.PE_B_dummy_U0.ap_ready | AESL_inst_kernel0.PE_B_dummy_U0.ap_idle))) begin
                                $display("//      Deadlock detected: can be a false alarm due to leftover data,");
                                $display("//      please try cosim_design -disable_deadlock_detection");
                            end
                        end
                    end
                    endcase
                end
                15 : begin
                    case(index2)
                    16: begin
                        if (~AESL_inst_kernel0.C_drain_IO_L1_out_he_U0.grp_C_drain_IO_L1_out_in_1_fu_113.fifo_C_drain_out_V_V_blk_n) begin
                            chan_path = "kernel0.fifo_C_drain_C_drain_6_U";
                            if (~AESL_inst_kernel0.fifo_C_drain_C_drain_6_U.if_empty_n) begin
                                $display("//      Channel: %0s, EMPTY", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~AESL_inst_kernel0.fifo_C_drain_C_drain_6_U.if_full_n) begin
                                $display("//      Channel: %0s, FULL", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                            else begin
                                $display("//      Channel: %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status NULL");
                            end
                        end
                    end
                    7: begin
                        if (~AESL_inst_kernel0.C_drain_IO_L1_out_he_U0.grp_C_drain_IO_L1_out_in_fu_106.fifo_C_drain_local_in_V_blk_n) begin
                            chan_path = "kernel0.fifo_C_drain_PE_0_0_s_U";
                            if (~AESL_inst_kernel0.fifo_C_drain_PE_0_0_s_U.if_empty_n) begin
                                $display("//      Channel: %0s, EMPTY", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~AESL_inst_kernel0.fifo_C_drain_PE_0_0_s_U.if_full_n) begin
                                $display("//      Channel: %0s, FULL", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                            else begin
                                $display("//      Channel: %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status NULL");
                            end
                        end
                        if ((~AESL_inst_kernel0.start_for_C_drainwdI_U.if_empty_n & (AESL_inst_kernel0.C_drain_IO_L1_out_he_U0.ap_ready | AESL_inst_kernel0.C_drain_IO_L1_out_he_U0.ap_idle))) begin
                            chan_path = "";
                            if ((~AESL_inst_kernel0.start_for_C_drainwdI_U.if_empty_n & (AESL_inst_kernel0.C_drain_IO_L1_out_he_U0.ap_ready | AESL_inst_kernel0.C_drain_IO_L1_out_he_U0.ap_idle))) begin
                                $display("//      Deadlock detected: can be a false alarm due to leftover data,");
                                $display("//      please try cosim_design -disable_deadlock_detection");
                            end
                        end
                    end
                    endcase
                end
                16 : begin
                    case(index2)
                    15: begin
                        if (~AESL_inst_kernel0.C_drain_IO_L1_out145_U0.grp_C_drain_IO_L1_out_in_2_fu_120.fifo_C_drain_in_V_V_blk_n) begin
                            chan_path = "kernel0.fifo_C_drain_C_drain_6_U";
                            if (~AESL_inst_kernel0.fifo_C_drain_C_drain_6_U.if_empty_n) begin
                                $display("//      Channel: %0s, EMPTY", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~AESL_inst_kernel0.fifo_C_drain_C_drain_6_U.if_full_n) begin
                                $display("//      Channel: %0s, FULL", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                            else begin
                                $display("//      Channel: %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status NULL");
                            end
                        end
                    end
                    19: begin
                        if (~AESL_inst_kernel0.C_drain_IO_L1_out145_U0.grp_C_drain_IO_L1_out_in_2_fu_120.fifo_C_drain_out_V_V_blk_n) begin
                            chan_path = "kernel0.fifo_C_drain_C_drain_7_U";
                            if (~AESL_inst_kernel0.fifo_C_drain_C_drain_7_U.if_empty_n) begin
                                $display("//      Channel: %0s, EMPTY", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~AESL_inst_kernel0.fifo_C_drain_C_drain_7_U.if_full_n) begin
                                $display("//      Channel: %0s, FULL", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                            else begin
                                $display("//      Channel: %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status NULL");
                            end
                        end
                        if ((~AESL_inst_kernel0.start_for_C_drainEe0_U.if_full_n & AESL_inst_kernel0.C_drain_IO_L2_out_he_U0.ap_done)) begin
                            chan_path = "";
                            if ((~AESL_inst_kernel0.start_for_C_drainEe0_U.if_full_n & AESL_inst_kernel0.C_drain_IO_L2_out_he_U0.ap_done)) begin
                                $display("//      Deadlock detected: can be a false alarm due to leftover data,");
                                $display("//      please try cosim_design -disable_deadlock_detection");
                            end
                        end
                    end
                    10: begin
                        if (~AESL_inst_kernel0.C_drain_IO_L1_out145_U0.grp_C_drain_IO_L1_out_in_fu_130.fifo_C_drain_local_in_V_blk_n) begin
                            chan_path = "kernel0.fifo_C_drain_PE_1_0_s_U";
                            if (~AESL_inst_kernel0.fifo_C_drain_PE_1_0_s_U.if_empty_n) begin
                                $display("//      Channel: %0s, EMPTY", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~AESL_inst_kernel0.fifo_C_drain_PE_1_0_s_U.if_full_n) begin
                                $display("//      Channel: %0s, FULL", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                            else begin
                                $display("//      Channel: %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status NULL");
                            end
                        end
                        if ((~AESL_inst_kernel0.start_for_C_drainAem_U.if_empty_n & (AESL_inst_kernel0.C_drain_IO_L1_out145_U0.ap_ready | AESL_inst_kernel0.C_drain_IO_L1_out145_U0.ap_idle))) begin
                            chan_path = "";
                            if ((~AESL_inst_kernel0.start_for_C_drainAem_U.if_empty_n & (AESL_inst_kernel0.C_drain_IO_L1_out145_U0.ap_ready | AESL_inst_kernel0.C_drain_IO_L1_out145_U0.ap_idle))) begin
                                $display("//      Deadlock detected: can be a false alarm due to leftover data,");
                                $display("//      please try cosim_design -disable_deadlock_detection");
                            end
                        end
                    end
                    endcase
                end
                17 : begin
                    case(index2)
                    18: begin
                        if (~AESL_inst_kernel0.C_drain_IO_L1_out_he_1_U0.grp_C_drain_IO_L1_out_in_1_fu_113.fifo_C_drain_out_V_V_blk_n) begin
                            chan_path = "kernel0.fifo_C_drain_C_drain_8_U";
                            if (~AESL_inst_kernel0.fifo_C_drain_C_drain_8_U.if_empty_n) begin
                                $display("//      Channel: %0s, EMPTY", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~AESL_inst_kernel0.fifo_C_drain_C_drain_8_U.if_full_n) begin
                                $display("//      Channel: %0s, FULL", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                            else begin
                                $display("//      Channel: %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status NULL");
                            end
                        end
                    end
                    8: begin
                        if (~AESL_inst_kernel0.C_drain_IO_L1_out_he_1_U0.grp_C_drain_IO_L1_out_in_fu_106.fifo_C_drain_local_in_V_blk_n) begin
                            chan_path = "kernel0.fifo_C_drain_PE_0_1_s_U";
                            if (~AESL_inst_kernel0.fifo_C_drain_PE_0_1_s_U.if_empty_n) begin
                                $display("//      Channel: %0s, EMPTY", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~AESL_inst_kernel0.fifo_C_drain_PE_0_1_s_U.if_full_n) begin
                                $display("//      Channel: %0s, FULL", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                            else begin
                                $display("//      Channel: %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status NULL");
                            end
                        end
                        if ((~AESL_inst_kernel0.start_for_C_drainyd2_U.if_empty_n & (AESL_inst_kernel0.C_drain_IO_L1_out_he_1_U0.ap_ready | AESL_inst_kernel0.C_drain_IO_L1_out_he_1_U0.ap_idle))) begin
                            chan_path = "";
                            if ((~AESL_inst_kernel0.start_for_C_drainyd2_U.if_empty_n & (AESL_inst_kernel0.C_drain_IO_L1_out_he_1_U0.ap_ready | AESL_inst_kernel0.C_drain_IO_L1_out_he_1_U0.ap_idle))) begin
                                $display("//      Deadlock detected: can be a false alarm due to leftover data,");
                                $display("//      please try cosim_design -disable_deadlock_detection");
                            end
                        end
                    end
                    endcase
                end
                18 : begin
                    case(index2)
                    17: begin
                        if (~AESL_inst_kernel0.C_drain_IO_L1_out_U0.grp_C_drain_IO_L1_out_in_2_fu_120.fifo_C_drain_in_V_V_blk_n) begin
                            chan_path = "kernel0.fifo_C_drain_C_drain_8_U";
                            if (~AESL_inst_kernel0.fifo_C_drain_C_drain_8_U.if_empty_n) begin
                                $display("//      Channel: %0s, EMPTY", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~AESL_inst_kernel0.fifo_C_drain_C_drain_8_U.if_full_n) begin
                                $display("//      Channel: %0s, FULL", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                            else begin
                                $display("//      Channel: %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status NULL");
                            end
                        end
                    end
                    20: begin
                        if (~AESL_inst_kernel0.C_drain_IO_L1_out_U0.grp_C_drain_IO_L1_out_in_2_fu_120.fifo_C_drain_out_V_V_blk_n) begin
                            chan_path = "kernel0.fifo_C_drain_C_drain_9_U";
                            if (~AESL_inst_kernel0.fifo_C_drain_C_drain_9_U.if_empty_n) begin
                                $display("//      Channel: %0s, EMPTY", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~AESL_inst_kernel0.fifo_C_drain_C_drain_9_U.if_full_n) begin
                                $display("//      Channel: %0s, FULL", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                            else begin
                                $display("//      Channel: %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status NULL");
                            end
                        end
                        if ((~AESL_inst_kernel0.start_for_C_drainFfa_U.if_full_n & AESL_inst_kernel0.C_drain_IO_L2_out_U0.ap_done)) begin
                            chan_path = "";
                            if ((~AESL_inst_kernel0.start_for_C_drainFfa_U.if_full_n & AESL_inst_kernel0.C_drain_IO_L2_out_U0.ap_done)) begin
                                $display("//      Deadlock detected: can be a false alarm due to leftover data,");
                                $display("//      please try cosim_design -disable_deadlock_detection");
                            end
                        end
                    end
                    12: begin
                        if (~AESL_inst_kernel0.C_drain_IO_L1_out_U0.grp_C_drain_IO_L1_out_in_fu_130.fifo_C_drain_local_in_V_blk_n) begin
                            chan_path = "kernel0.fifo_C_drain_PE_1_1_s_U";
                            if (~AESL_inst_kernel0.fifo_C_drain_PE_1_1_s_U.if_empty_n) begin
                                $display("//      Channel: %0s, EMPTY", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~AESL_inst_kernel0.fifo_C_drain_PE_1_1_s_U.if_full_n) begin
                                $display("//      Channel: %0s, FULL", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                            else begin
                                $display("//      Channel: %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status NULL");
                            end
                        end
                        if ((~AESL_inst_kernel0.start_for_C_drainDeQ_U.if_empty_n & (AESL_inst_kernel0.C_drain_IO_L1_out_U0.ap_ready | AESL_inst_kernel0.C_drain_IO_L1_out_U0.ap_idle))) begin
                            chan_path = "";
                            if ((~AESL_inst_kernel0.start_for_C_drainDeQ_U.if_empty_n & (AESL_inst_kernel0.C_drain_IO_L1_out_U0.ap_ready | AESL_inst_kernel0.C_drain_IO_L1_out_U0.ap_idle))) begin
                                $display("//      Deadlock detected: can be a false alarm due to leftover data,");
                                $display("//      please try cosim_design -disable_deadlock_detection");
                            end
                        end
                    end
                    endcase
                end
                19 : begin
                    case(index2)
                    20: begin
                        if (~AESL_inst_kernel0.C_drain_IO_L2_out_he_U0.fifo_C_drain_out_V_V_blk_n) begin
                            chan_path = "kernel0.fifo_C_drain_C_drain_10_U";
                            if (~AESL_inst_kernel0.fifo_C_drain_C_drain_10_U.if_empty_n) begin
                                $display("//      Channel: %0s, EMPTY", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~AESL_inst_kernel0.fifo_C_drain_C_drain_10_U.if_full_n) begin
                                $display("//      Channel: %0s, FULL", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                            else begin
                                $display("//      Channel: %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status NULL");
                            end
                        end
                    end
                    16: begin
                        if (~AESL_inst_kernel0.C_drain_IO_L2_out_he_U0.fifo_C_drain_local_in_V_V_blk_n) begin
                            chan_path = "kernel0.fifo_C_drain_C_drain_7_U";
                            if (~AESL_inst_kernel0.fifo_C_drain_C_drain_7_U.if_empty_n) begin
                                $display("//      Channel: %0s, EMPTY", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~AESL_inst_kernel0.fifo_C_drain_C_drain_7_U.if_full_n) begin
                                $display("//      Channel: %0s, FULL", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                            else begin
                                $display("//      Channel: %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status NULL");
                            end
                        end
                        if ((~AESL_inst_kernel0.start_for_C_drainEe0_U.if_empty_n & (AESL_inst_kernel0.C_drain_IO_L2_out_he_U0.ap_ready | AESL_inst_kernel0.C_drain_IO_L2_out_he_U0.ap_idle))) begin
                            chan_path = "";
                            if ((~AESL_inst_kernel0.start_for_C_drainEe0_U.if_empty_n & (AESL_inst_kernel0.C_drain_IO_L2_out_he_U0.ap_ready | AESL_inst_kernel0.C_drain_IO_L2_out_he_U0.ap_idle))) begin
                                $display("//      Deadlock detected: can be a false alarm due to leftover data,");
                                $display("//      please try cosim_design -disable_deadlock_detection");
                            end
                        end
                    end
                    endcase
                end
                20 : begin
                    case(index2)
                    19: begin
                        if (~AESL_inst_kernel0.C_drain_IO_L2_out_U0.fifo_C_drain_in_V_V_blk_n) begin
                            chan_path = "kernel0.fifo_C_drain_C_drain_10_U";
                            if (~AESL_inst_kernel0.fifo_C_drain_C_drain_10_U.if_empty_n) begin
                                $display("//      Channel: %0s, EMPTY", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~AESL_inst_kernel0.fifo_C_drain_C_drain_10_U.if_full_n) begin
                                $display("//      Channel: %0s, FULL", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                            else begin
                                $display("//      Channel: %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status NULL");
                            end
                        end
                    end
                    21: begin
                        if (~AESL_inst_kernel0.C_drain_IO_L2_out_U0.fifo_C_drain_out_V_V_blk_n) begin
                            chan_path = "kernel0.fifo_C_drain_C_drain_11_U";
                            if (~AESL_inst_kernel0.fifo_C_drain_C_drain_11_U.if_empty_n) begin
                                $display("//      Channel: %0s, EMPTY", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~AESL_inst_kernel0.fifo_C_drain_C_drain_11_U.if_full_n) begin
                                $display("//      Channel: %0s, FULL", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                            else begin
                                $display("//      Channel: %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status NULL");
                            end
                        end
                    end
                    18: begin
                        if (~AESL_inst_kernel0.C_drain_IO_L2_out_U0.fifo_C_drain_local_in_V_V_blk_n) begin
                            chan_path = "kernel0.fifo_C_drain_C_drain_9_U";
                            if (~AESL_inst_kernel0.fifo_C_drain_C_drain_9_U.if_empty_n) begin
                                $display("//      Channel: %0s, EMPTY", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~AESL_inst_kernel0.fifo_C_drain_C_drain_9_U.if_full_n) begin
                                $display("//      Channel: %0s, FULL", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                            else begin
                                $display("//      Channel: %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status NULL");
                            end
                        end
                        if ((~AESL_inst_kernel0.start_for_C_drainFfa_U.if_empty_n & (AESL_inst_kernel0.C_drain_IO_L2_out_U0.ap_ready | AESL_inst_kernel0.C_drain_IO_L2_out_U0.ap_idle))) begin
                            chan_path = "";
                            if ((~AESL_inst_kernel0.start_for_C_drainFfa_U.if_empty_n & (AESL_inst_kernel0.C_drain_IO_L2_out_U0.ap_ready | AESL_inst_kernel0.C_drain_IO_L2_out_U0.ap_idle))) begin
                                $display("//      Deadlock detected: can be a false alarm due to leftover data,");
                                $display("//      please try cosim_design -disable_deadlock_detection");
                            end
                        end
                    end
                    endcase
                end
                21 : begin
                    case(index2)
                    0: begin
                        if (~AESL_inst_kernel0.C_drain_IO_L3_out_U0.C_V_offset_blk_n) begin
                            chan_path = "kernel0.C_V_c_U";
                            if (~AESL_inst_kernel0.C_V_c_U.if_empty_n) begin
                                $display("//      Channel: %0s, EMPTY", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~AESL_inst_kernel0.C_V_c_U.if_full_n) begin
                                $display("//      Channel: %0s, FULL", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                            else begin
                                $display("//      Channel: %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status NULL");
                            end
                        end
                        if ((~AESL_inst_kernel0.start_for_C_drainrcU_U.if_empty_n & (AESL_inst_kernel0.C_drain_IO_L3_out_U0.ap_ready | AESL_inst_kernel0.C_drain_IO_L3_out_U0.ap_idle))) begin
                            chan_path = "";
                            if ((~AESL_inst_kernel0.start_for_C_drainrcU_U.if_empty_n & (AESL_inst_kernel0.C_drain_IO_L3_out_U0.ap_ready | AESL_inst_kernel0.C_drain_IO_L3_out_U0.ap_idle))) begin
                                $display("//      Deadlock detected: can be a false alarm due to leftover data,");
                                $display("//      please try cosim_design -disable_deadlock_detection");
                            end
                        end
                    end
                    20: begin
                        if (~AESL_inst_kernel0.C_drain_IO_L3_out_U0.fifo_C_drain_local_in_V_V_blk_n) begin
                            chan_path = "kernel0.fifo_C_drain_C_drain_11_U";
                            if (~AESL_inst_kernel0.fifo_C_drain_C_drain_11_U.if_empty_n) begin
                                $display("//      Channel: %0s, EMPTY", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~AESL_inst_kernel0.fifo_C_drain_C_drain_11_U.if_full_n) begin
                                $display("//      Channel: %0s, FULL", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                            else begin
                                $display("//      Channel: %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_path %0s", chan_path);
                                $fdisplay(fp, "Dependence_Channel_status NULL");
                            end
                        end
                    end
                    endcase
                end
            endcase
        end
    endtask

    // report
    initial begin : report_deadlock
        integer cycle_id;
        integer cycle_comp_id;
        wait (reset == 1);
        cycle_id = 1;
        while (1) begin
            @ (negedge clock);
            case (CS_fsm)
                ST_DL_DETECTED: begin
                    cycle_comp_id = 2;
                    if (dl_detect_reg != dl_done_reg) begin
                        if (dl_done_reg == 'b0) begin
                            print_dl_head;
                        end
                        print_cycle_start(proc_path(origin), cycle_id);
                        cycle_id = cycle_id + 1;
                    end
                    else begin
                        print_dl_end(cycle_id - 1);
                        $finish;
                    end
                end
                ST_DL_REPORT: begin
                    if ((|(dl_in_vec)) & ~(|(dl_in_vec & origin_reg))) begin
                        print_cycle_chan_comp(dl_in_vec_reg, dl_in_vec);
                        print_cycle_proc_comp(proc_path(dl_in_vec), cycle_comp_id);
                        cycle_comp_id = cycle_comp_id + 1;
                    end
                    else begin
                        print_cycle_chan_comp(dl_in_vec_reg, dl_in_vec);
                    end
                end
            endcase
        end
    end
 
endmodule
