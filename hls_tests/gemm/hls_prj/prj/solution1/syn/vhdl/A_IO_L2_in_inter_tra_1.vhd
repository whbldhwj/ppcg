-- ==============================================================
-- RTL generated by Vivado(TM) HLS - High-Level Synthesis from C, C++ and OpenCL
-- Version: 2019.2
-- Copyright (C) 1986-2019 Xilinx, Inc. All Rights Reserved.
-- 
-- ===========================================================

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

entity A_IO_L2_in_inter_tra_1 is
port (
    ap_clk : IN STD_LOGIC;
    ap_rst : IN STD_LOGIC;
    ap_start : IN STD_LOGIC;
    ap_done : OUT STD_LOGIC;
    ap_idle : OUT STD_LOGIC;
    ap_ready : OUT STD_LOGIC;
    local_A_0_V_address0 : OUT STD_LOGIC_VECTOR (0 downto 0);
    local_A_0_V_ce0 : OUT STD_LOGIC;
    local_A_0_V_we0 : OUT STD_LOGIC;
    local_A_0_V_d0 : OUT STD_LOGIC_VECTOR (127 downto 0);
    fifo_A_in_V_V_dout : IN STD_LOGIC_VECTOR (127 downto 0);
    fifo_A_in_V_V_empty_n : IN STD_LOGIC;
    fifo_A_in_V_V_read : OUT STD_LOGIC;
    fifo_A_out_V_V_din : OUT STD_LOGIC_VECTOR (127 downto 0);
    fifo_A_out_V_V_full_n : IN STD_LOGIC;
    fifo_A_out_V_V_write : OUT STD_LOGIC );
end;


architecture behav of A_IO_L2_in_inter_tra_1 is 
    constant ap_const_logic_1 : STD_LOGIC := '1';
    constant ap_const_logic_0 : STD_LOGIC := '0';
    constant ap_ST_fsm_state1 : STD_LOGIC_VECTOR (2 downto 0) := "001";
    constant ap_ST_fsm_pp0_stage0 : STD_LOGIC_VECTOR (2 downto 0) := "010";
    constant ap_ST_fsm_state4 : STD_LOGIC_VECTOR (2 downto 0) := "100";
    constant ap_const_boolean_1 : BOOLEAN := true;
    constant ap_const_lv32_0 : STD_LOGIC_VECTOR (31 downto 0) := "00000000000000000000000000000000";
    constant ap_const_lv32_1 : STD_LOGIC_VECTOR (31 downto 0) := "00000000000000000000000000000001";
    constant ap_const_boolean_0 : BOOLEAN := false;
    constant ap_const_lv1_0 : STD_LOGIC_VECTOR (0 downto 0) := "0";
    constant ap_const_lv1_1 : STD_LOGIC_VECTOR (0 downto 0) := "1";
    constant ap_const_lv3_0 : STD_LOGIC_VECTOR (2 downto 0) := "000";
    constant ap_const_lv2_0 : STD_LOGIC_VECTOR (1 downto 0) := "00";
    constant ap_const_lv3_4 : STD_LOGIC_VECTOR (2 downto 0) := "100";
    constant ap_const_lv3_1 : STD_LOGIC_VECTOR (2 downto 0) := "001";
    constant ap_const_lv2_1 : STD_LOGIC_VECTOR (1 downto 0) := "01";
    constant ap_const_lv2_2 : STD_LOGIC_VECTOR (1 downto 0) := "10";
    constant ap_const_lv32_2 : STD_LOGIC_VECTOR (31 downto 0) := "00000000000000000000000000000010";

    signal ap_CS_fsm : STD_LOGIC_VECTOR (2 downto 0) := "001";
    attribute fsm_encoding : string;
    attribute fsm_encoding of ap_CS_fsm : signal is "none";
    signal ap_CS_fsm_state1 : STD_LOGIC;
    attribute fsm_encoding of ap_CS_fsm_state1 : signal is "none";
    signal fifo_A_in_V_V_blk_n : STD_LOGIC;
    signal ap_CS_fsm_pp0_stage0 : STD_LOGIC;
    attribute fsm_encoding of ap_CS_fsm_pp0_stage0 : signal is "none";
    signal ap_enable_reg_pp0_iter1 : STD_LOGIC := '0';
    signal ap_block_pp0_stage0 : BOOLEAN;
    signal icmp_ln75_reg_205 : STD_LOGIC_VECTOR (0 downto 0);
    signal fifo_A_out_V_V_blk_n : STD_LOGIC;
    signal select_ln82_1_reg_219 : STD_LOGIC_VECTOR (0 downto 0);
    signal indvar_flatten_reg_102 : STD_LOGIC_VECTOR (2 downto 0);
    signal c3_0_reg_113 : STD_LOGIC_VECTOR (1 downto 0);
    signal c4_0_reg_124 : STD_LOGIC_VECTOR (1 downto 0);
    signal icmp_ln75_fu_135_p2 : STD_LOGIC_VECTOR (0 downto 0);
    signal ap_block_state2_pp0_stage0_iter0 : BOOLEAN;
    signal ap_block_state3_pp0_stage0_iter1 : BOOLEAN;
    signal ap_block_pp0_stage0_11001 : BOOLEAN;
    signal add_ln75_fu_141_p2 : STD_LOGIC_VECTOR (2 downto 0);
    signal ap_enable_reg_pp0_iter0 : STD_LOGIC := '0';
    signal select_ln82_fu_159_p3 : STD_LOGIC_VECTOR (1 downto 0);
    signal select_ln82_reg_214 : STD_LOGIC_VECTOR (1 downto 0);
    signal select_ln82_1_fu_179_p3 : STD_LOGIC_VECTOR (0 downto 0);
    signal select_ln75_fu_187_p3 : STD_LOGIC_VECTOR (1 downto 0);
    signal c4_fu_195_p2 : STD_LOGIC_VECTOR (1 downto 0);
    signal ap_block_pp0_stage0_subdone : BOOLEAN;
    signal ap_condition_pp0_exit_iter0_state2 : STD_LOGIC;
    signal zext_ln83_fu_201_p1 : STD_LOGIC_VECTOR (63 downto 0);
    signal ap_block_pp0_stage0_01001 : BOOLEAN;
    signal icmp_ln77_fu_153_p2 : STD_LOGIC_VECTOR (0 downto 0);
    signal c3_fu_147_p2 : STD_LOGIC_VECTOR (1 downto 0);
    signal icmp_ln82_fu_167_p2 : STD_LOGIC_VECTOR (0 downto 0);
    signal icmp_ln82_1_fu_173_p2 : STD_LOGIC_VECTOR (0 downto 0);
    signal ap_CS_fsm_state4 : STD_LOGIC;
    attribute fsm_encoding of ap_CS_fsm_state4 : signal is "none";
    signal ap_NS_fsm : STD_LOGIC_VECTOR (2 downto 0);
    signal ap_idle_pp0 : STD_LOGIC;
    signal ap_enable_pp0 : STD_LOGIC;


begin




    ap_CS_fsm_assign_proc : process(ap_clk)
    begin
        if (ap_clk'event and ap_clk =  '1') then
            if (ap_rst = '1') then
                ap_CS_fsm <= ap_ST_fsm_state1;
            else
                ap_CS_fsm <= ap_NS_fsm;
            end if;
        end if;
    end process;


    ap_enable_reg_pp0_iter0_assign_proc : process(ap_clk)
    begin
        if (ap_clk'event and ap_clk =  '1') then
            if (ap_rst = '1') then
                ap_enable_reg_pp0_iter0 <= ap_const_logic_0;
            else
                if (((ap_const_boolean_0 = ap_block_pp0_stage0_subdone) and (ap_const_logic_1 = ap_CS_fsm_pp0_stage0) and (ap_const_logic_1 = ap_condition_pp0_exit_iter0_state2))) then 
                    ap_enable_reg_pp0_iter0 <= ap_const_logic_0;
                elsif (((ap_start = ap_const_logic_1) and (ap_const_logic_1 = ap_CS_fsm_state1))) then 
                    ap_enable_reg_pp0_iter0 <= ap_const_logic_1;
                end if; 
            end if;
        end if;
    end process;


    ap_enable_reg_pp0_iter1_assign_proc : process(ap_clk)
    begin
        if (ap_clk'event and ap_clk =  '1') then
            if (ap_rst = '1') then
                ap_enable_reg_pp0_iter1 <= ap_const_logic_0;
            else
                if (((ap_const_boolean_0 = ap_block_pp0_stage0_subdone) and (ap_const_logic_1 = ap_condition_pp0_exit_iter0_state2))) then 
                    ap_enable_reg_pp0_iter1 <= (ap_const_logic_1 xor ap_condition_pp0_exit_iter0_state2);
                elsif ((ap_const_boolean_0 = ap_block_pp0_stage0_subdone)) then 
                    ap_enable_reg_pp0_iter1 <= ap_enable_reg_pp0_iter0;
                elsif (((ap_start = ap_const_logic_1) and (ap_const_logic_1 = ap_CS_fsm_state1))) then 
                    ap_enable_reg_pp0_iter1 <= ap_const_logic_0;
                end if; 
            end if;
        end if;
    end process;


    c3_0_reg_113_assign_proc : process (ap_clk)
    begin
        if (ap_clk'event and ap_clk = '1') then
            if (((ap_const_boolean_0 = ap_block_pp0_stage0_11001) and (icmp_ln75_fu_135_p2 = ap_const_lv1_0) and (ap_enable_reg_pp0_iter0 = ap_const_logic_1) and (ap_const_logic_1 = ap_CS_fsm_pp0_stage0))) then 
                c3_0_reg_113 <= select_ln75_fu_187_p3;
            elsif (((ap_start = ap_const_logic_1) and (ap_const_logic_1 = ap_CS_fsm_state1))) then 
                c3_0_reg_113 <= ap_const_lv2_0;
            end if; 
        end if;
    end process;

    c4_0_reg_124_assign_proc : process (ap_clk)
    begin
        if (ap_clk'event and ap_clk = '1') then
            if (((ap_const_boolean_0 = ap_block_pp0_stage0_11001) and (icmp_ln75_fu_135_p2 = ap_const_lv1_0) and (ap_enable_reg_pp0_iter0 = ap_const_logic_1) and (ap_const_logic_1 = ap_CS_fsm_pp0_stage0))) then 
                c4_0_reg_124 <= c4_fu_195_p2;
            elsif (((ap_start = ap_const_logic_1) and (ap_const_logic_1 = ap_CS_fsm_state1))) then 
                c4_0_reg_124 <= ap_const_lv2_0;
            end if; 
        end if;
    end process;

    indvar_flatten_reg_102_assign_proc : process (ap_clk)
    begin
        if (ap_clk'event and ap_clk = '1') then
            if (((ap_const_boolean_0 = ap_block_pp0_stage0_11001) and (icmp_ln75_fu_135_p2 = ap_const_lv1_0) and (ap_enable_reg_pp0_iter0 = ap_const_logic_1) and (ap_const_logic_1 = ap_CS_fsm_pp0_stage0))) then 
                indvar_flatten_reg_102 <= add_ln75_fu_141_p2;
            elsif (((ap_start = ap_const_logic_1) and (ap_const_logic_1 = ap_CS_fsm_state1))) then 
                indvar_flatten_reg_102 <= ap_const_lv3_0;
            end if; 
        end if;
    end process;
    process (ap_clk)
    begin
        if (ap_clk'event and ap_clk = '1') then
            if (((ap_const_boolean_0 = ap_block_pp0_stage0_11001) and (ap_const_logic_1 = ap_CS_fsm_pp0_stage0))) then
                icmp_ln75_reg_205 <= icmp_ln75_fu_135_p2;
            end if;
        end if;
    end process;
    process (ap_clk)
    begin
        if (ap_clk'event and ap_clk = '1') then
            if (((ap_const_boolean_0 = ap_block_pp0_stage0_11001) and (icmp_ln75_fu_135_p2 = ap_const_lv1_0) and (ap_const_logic_1 = ap_CS_fsm_pp0_stage0))) then
                select_ln82_1_reg_219 <= select_ln82_1_fu_179_p3;
                select_ln82_reg_214 <= select_ln82_fu_159_p3;
            end if;
        end if;
    end process;

    ap_NS_fsm_assign_proc : process (ap_start, ap_CS_fsm, ap_CS_fsm_state1, icmp_ln75_fu_135_p2, ap_enable_reg_pp0_iter0, ap_block_pp0_stage0_subdone)
    begin
        case ap_CS_fsm is
            when ap_ST_fsm_state1 => 
                if (((ap_start = ap_const_logic_1) and (ap_const_logic_1 = ap_CS_fsm_state1))) then
                    ap_NS_fsm <= ap_ST_fsm_pp0_stage0;
                else
                    ap_NS_fsm <= ap_ST_fsm_state1;
                end if;
            when ap_ST_fsm_pp0_stage0 => 
                if (not(((ap_const_boolean_0 = ap_block_pp0_stage0_subdone) and (icmp_ln75_fu_135_p2 = ap_const_lv1_1) and (ap_enable_reg_pp0_iter0 = ap_const_logic_1)))) then
                    ap_NS_fsm <= ap_ST_fsm_pp0_stage0;
                elsif (((ap_const_boolean_0 = ap_block_pp0_stage0_subdone) and (icmp_ln75_fu_135_p2 = ap_const_lv1_1) and (ap_enable_reg_pp0_iter0 = ap_const_logic_1))) then
                    ap_NS_fsm <= ap_ST_fsm_state4;
                else
                    ap_NS_fsm <= ap_ST_fsm_pp0_stage0;
                end if;
            when ap_ST_fsm_state4 => 
                ap_NS_fsm <= ap_ST_fsm_state1;
            when others =>  
                ap_NS_fsm <= "XXX";
        end case;
    end process;
    add_ln75_fu_141_p2 <= std_logic_vector(unsigned(indvar_flatten_reg_102) + unsigned(ap_const_lv3_1));
    ap_CS_fsm_pp0_stage0 <= ap_CS_fsm(1);
    ap_CS_fsm_state1 <= ap_CS_fsm(0);
    ap_CS_fsm_state4 <= ap_CS_fsm(2);
        ap_block_pp0_stage0 <= not((ap_const_boolean_1 = ap_const_boolean_1));

    ap_block_pp0_stage0_01001_assign_proc : process(fifo_A_in_V_V_empty_n, fifo_A_out_V_V_full_n, ap_enable_reg_pp0_iter1, icmp_ln75_reg_205, select_ln82_1_reg_219)
    begin
                ap_block_pp0_stage0_01001 <= ((ap_enable_reg_pp0_iter1 = ap_const_logic_1) and (((select_ln82_1_reg_219 = ap_const_lv1_0) and (fifo_A_out_V_V_full_n = ap_const_logic_0)) or ((icmp_ln75_reg_205 = ap_const_lv1_0) and (fifo_A_in_V_V_empty_n = ap_const_logic_0))));
    end process;


    ap_block_pp0_stage0_11001_assign_proc : process(fifo_A_in_V_V_empty_n, fifo_A_out_V_V_full_n, ap_enable_reg_pp0_iter1, icmp_ln75_reg_205, select_ln82_1_reg_219)
    begin
                ap_block_pp0_stage0_11001 <= ((ap_enable_reg_pp0_iter1 = ap_const_logic_1) and (((select_ln82_1_reg_219 = ap_const_lv1_0) and (fifo_A_out_V_V_full_n = ap_const_logic_0)) or ((icmp_ln75_reg_205 = ap_const_lv1_0) and (fifo_A_in_V_V_empty_n = ap_const_logic_0))));
    end process;


    ap_block_pp0_stage0_subdone_assign_proc : process(fifo_A_in_V_V_empty_n, fifo_A_out_V_V_full_n, ap_enable_reg_pp0_iter1, icmp_ln75_reg_205, select_ln82_1_reg_219)
    begin
                ap_block_pp0_stage0_subdone <= ((ap_enable_reg_pp0_iter1 = ap_const_logic_1) and (((select_ln82_1_reg_219 = ap_const_lv1_0) and (fifo_A_out_V_V_full_n = ap_const_logic_0)) or ((icmp_ln75_reg_205 = ap_const_lv1_0) and (fifo_A_in_V_V_empty_n = ap_const_logic_0))));
    end process;

        ap_block_state2_pp0_stage0_iter0 <= not((ap_const_boolean_1 = ap_const_boolean_1));

    ap_block_state3_pp0_stage0_iter1_assign_proc : process(fifo_A_in_V_V_empty_n, fifo_A_out_V_V_full_n, icmp_ln75_reg_205, select_ln82_1_reg_219)
    begin
                ap_block_state3_pp0_stage0_iter1 <= (((select_ln82_1_reg_219 = ap_const_lv1_0) and (fifo_A_out_V_V_full_n = ap_const_logic_0)) or ((icmp_ln75_reg_205 = ap_const_lv1_0) and (fifo_A_in_V_V_empty_n = ap_const_logic_0)));
    end process;


    ap_condition_pp0_exit_iter0_state2_assign_proc : process(icmp_ln75_fu_135_p2)
    begin
        if ((icmp_ln75_fu_135_p2 = ap_const_lv1_1)) then 
            ap_condition_pp0_exit_iter0_state2 <= ap_const_logic_1;
        else 
            ap_condition_pp0_exit_iter0_state2 <= ap_const_logic_0;
        end if; 
    end process;


    ap_done_assign_proc : process(ap_start, ap_CS_fsm_state1, ap_CS_fsm_state4)
    begin
        if (((ap_const_logic_1 = ap_CS_fsm_state4) or ((ap_start = ap_const_logic_0) and (ap_const_logic_1 = ap_CS_fsm_state1)))) then 
            ap_done <= ap_const_logic_1;
        else 
            ap_done <= ap_const_logic_0;
        end if; 
    end process;

    ap_enable_pp0 <= (ap_idle_pp0 xor ap_const_logic_1);

    ap_idle_assign_proc : process(ap_start, ap_CS_fsm_state1)
    begin
        if (((ap_start = ap_const_logic_0) and (ap_const_logic_1 = ap_CS_fsm_state1))) then 
            ap_idle <= ap_const_logic_1;
        else 
            ap_idle <= ap_const_logic_0;
        end if; 
    end process;


    ap_idle_pp0_assign_proc : process(ap_enable_reg_pp0_iter1, ap_enable_reg_pp0_iter0)
    begin
        if (((ap_enable_reg_pp0_iter0 = ap_const_logic_0) and (ap_enable_reg_pp0_iter1 = ap_const_logic_0))) then 
            ap_idle_pp0 <= ap_const_logic_1;
        else 
            ap_idle_pp0 <= ap_const_logic_0;
        end if; 
    end process;


    ap_ready_assign_proc : process(ap_CS_fsm_state4)
    begin
        if ((ap_const_logic_1 = ap_CS_fsm_state4)) then 
            ap_ready <= ap_const_logic_1;
        else 
            ap_ready <= ap_const_logic_0;
        end if; 
    end process;

    c3_fu_147_p2 <= std_logic_vector(unsigned(c3_0_reg_113) + unsigned(ap_const_lv2_1));
    c4_fu_195_p2 <= std_logic_vector(unsigned(select_ln82_fu_159_p3) + unsigned(ap_const_lv2_1));

    fifo_A_in_V_V_blk_n_assign_proc : process(fifo_A_in_V_V_empty_n, ap_CS_fsm_pp0_stage0, ap_enable_reg_pp0_iter1, ap_block_pp0_stage0, icmp_ln75_reg_205)
    begin
        if (((icmp_ln75_reg_205 = ap_const_lv1_0) and (ap_const_boolean_0 = ap_block_pp0_stage0) and (ap_enable_reg_pp0_iter1 = ap_const_logic_1) and (ap_const_logic_1 = ap_CS_fsm_pp0_stage0))) then 
            fifo_A_in_V_V_blk_n <= fifo_A_in_V_V_empty_n;
        else 
            fifo_A_in_V_V_blk_n <= ap_const_logic_1;
        end if; 
    end process;


    fifo_A_in_V_V_read_assign_proc : process(ap_CS_fsm_pp0_stage0, ap_enable_reg_pp0_iter1, icmp_ln75_reg_205, ap_block_pp0_stage0_11001)
    begin
        if (((ap_const_boolean_0 = ap_block_pp0_stage0_11001) and (icmp_ln75_reg_205 = ap_const_lv1_0) and (ap_enable_reg_pp0_iter1 = ap_const_logic_1) and (ap_const_logic_1 = ap_CS_fsm_pp0_stage0))) then 
            fifo_A_in_V_V_read <= ap_const_logic_1;
        else 
            fifo_A_in_V_V_read <= ap_const_logic_0;
        end if; 
    end process;


    fifo_A_out_V_V_blk_n_assign_proc : process(fifo_A_out_V_V_full_n, ap_CS_fsm_pp0_stage0, ap_enable_reg_pp0_iter1, ap_block_pp0_stage0, select_ln82_1_reg_219)
    begin
        if (((select_ln82_1_reg_219 = ap_const_lv1_0) and (ap_const_boolean_0 = ap_block_pp0_stage0) and (ap_enable_reg_pp0_iter1 = ap_const_logic_1) and (ap_const_logic_1 = ap_CS_fsm_pp0_stage0))) then 
            fifo_A_out_V_V_blk_n <= fifo_A_out_V_V_full_n;
        else 
            fifo_A_out_V_V_blk_n <= ap_const_logic_1;
        end if; 
    end process;

    fifo_A_out_V_V_din <= fifo_A_in_V_V_dout;

    fifo_A_out_V_V_write_assign_proc : process(ap_CS_fsm_pp0_stage0, ap_enable_reg_pp0_iter1, select_ln82_1_reg_219, ap_block_pp0_stage0_11001)
    begin
        if (((ap_const_boolean_0 = ap_block_pp0_stage0_11001) and (select_ln82_1_reg_219 = ap_const_lv1_0) and (ap_enable_reg_pp0_iter1 = ap_const_logic_1) and (ap_const_logic_1 = ap_CS_fsm_pp0_stage0))) then 
            fifo_A_out_V_V_write <= ap_const_logic_1;
        else 
            fifo_A_out_V_V_write <= ap_const_logic_0;
        end if; 
    end process;

    icmp_ln75_fu_135_p2 <= "1" when (indvar_flatten_reg_102 = ap_const_lv3_4) else "0";
    icmp_ln77_fu_153_p2 <= "1" when (c4_0_reg_124 = ap_const_lv2_2) else "0";
    icmp_ln82_1_fu_173_p2 <= "1" when (c3_0_reg_113 = ap_const_lv2_0) else "0";
    icmp_ln82_fu_167_p2 <= "1" when (c3_fu_147_p2 = ap_const_lv2_0) else "0";
    local_A_0_V_address0 <= zext_ln83_fu_201_p1(1 - 1 downto 0);

    local_A_0_V_ce0_assign_proc : process(ap_CS_fsm_pp0_stage0, ap_enable_reg_pp0_iter1, ap_block_pp0_stage0_11001)
    begin
        if (((ap_const_boolean_0 = ap_block_pp0_stage0_11001) and (ap_enable_reg_pp0_iter1 = ap_const_logic_1) and (ap_const_logic_1 = ap_CS_fsm_pp0_stage0))) then 
            local_A_0_V_ce0 <= ap_const_logic_1;
        else 
            local_A_0_V_ce0 <= ap_const_logic_0;
        end if; 
    end process;

    local_A_0_V_d0 <= fifo_A_in_V_V_dout;

    local_A_0_V_we0_assign_proc : process(ap_CS_fsm_pp0_stage0, ap_enable_reg_pp0_iter1, select_ln82_1_reg_219, ap_block_pp0_stage0_11001)
    begin
        if (((ap_const_boolean_0 = ap_block_pp0_stage0_11001) and (select_ln82_1_reg_219 = ap_const_lv1_1) and (ap_enable_reg_pp0_iter1 = ap_const_logic_1) and (ap_const_logic_1 = ap_CS_fsm_pp0_stage0))) then 
            local_A_0_V_we0 <= ap_const_logic_1;
        else 
            local_A_0_V_we0 <= ap_const_logic_0;
        end if; 
    end process;

    select_ln75_fu_187_p3 <= 
        c3_fu_147_p2 when (icmp_ln77_fu_153_p2(0) = '1') else 
        c3_0_reg_113;
    select_ln82_1_fu_179_p3 <= 
        icmp_ln82_fu_167_p2 when (icmp_ln77_fu_153_p2(0) = '1') else 
        icmp_ln82_1_fu_173_p2;
    select_ln82_fu_159_p3 <= 
        ap_const_lv2_0 when (icmp_ln77_fu_153_p2(0) = '1') else 
        c4_0_reg_124;
    zext_ln83_fu_201_p1 <= std_logic_vector(IEEE.numeric_std.resize(unsigned(select_ln82_reg_214),64));
end behav;
