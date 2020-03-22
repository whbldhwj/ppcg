// ==============================================================
// Vivado(TM) HLS - High-Level Synthesis from C, C++ and SystemC v2019.2 (64-bit)
// Copyright 1986-2019 Xilinx, Inc. All Rights Reserved.
// ==============================================================

#define AP_INT_MAX_W 32678

#include <systemc>
#include <iostream>
#include <cstdlib>
#include <cstddef>
#include <stdint.h>
#include "SysCFileHandler.h"
#include "ap_int.h"
#include "ap_fixed.h"
#include <complex>
#include <stdbool.h>
#include "autopilot_cbe.h"
#include "hls_stream.h"
#include "hls_half.h"
#include "hls_signal_handler.h"

using namespace std;
using namespace sc_core;
using namespace sc_dt;


// [dump_struct_tree [build_nameSpaceTree] dumpedStructList] ---------->


// [dump_enumeration [get_enumeration_list]] ---------->


// wrapc file define: "gmem_A"
#define AUTOTB_TVIN_gmem_A  "../tv/cdatafile/c.kernel0.autotvin_gmem_A.dat"
// wrapc file define: "gmem_B"
#define AUTOTB_TVIN_gmem_B  "../tv/cdatafile/c.kernel0.autotvin_gmem_B.dat"
// wrapc file define: "gmem_C"
#define AUTOTB_TVOUT_gmem_C  "../tv/cdatafile/c.kernel0.autotvout_gmem_C.dat"
#define AUTOTB_TVIN_gmem_C  "../tv/cdatafile/c.kernel0.autotvin_gmem_C.dat"
// wrapc file define: "A_V"
#define AUTOTB_TVIN_A_V  "../tv/cdatafile/c.kernel0.autotvin_A_V.dat"
// wrapc file define: "B_V"
#define AUTOTB_TVIN_B_V  "../tv/cdatafile/c.kernel0.autotvin_B_V.dat"
// wrapc file define: "C_V"
#define AUTOTB_TVIN_C_V  "../tv/cdatafile/c.kernel0.autotvin_C_V.dat"

#define INTER_TCL  "../tv/cdatafile/ref.tcl"

// tvout file define: "gmem_C"
#define AUTOTB_TVOUT_PC_gmem_C  "../tv/rtldatafile/rtl.kernel0.autotvout_gmem_C.dat"

class INTER_TCL_FILE {
	public:
		INTER_TCL_FILE(const char* name) {
			mName = name;
			gmem_A_depth = 0;
			gmem_B_depth = 0;
			gmem_C_depth = 0;
			A_V_depth = 0;
			B_V_depth = 0;
			C_V_depth = 0;
			trans_num =0;
		}

		~INTER_TCL_FILE() {
			mFile.open(mName);
			if (!mFile.good()) {
				cout << "Failed to open file ref.tcl" << endl;
				exit (1);
			}
			string total_list = get_depth_list();
			mFile << "set depth_list {\n";
			mFile << total_list;
			mFile << "}\n";
			mFile << "set trans_num "<<trans_num<<endl;
			mFile.close();
		}

		string get_depth_list () {
			stringstream total_list;
			total_list << "{gmem_A " << gmem_A_depth << "}\n";
			total_list << "{gmem_B " << gmem_B_depth << "}\n";
			total_list << "{gmem_C " << gmem_C_depth << "}\n";
			total_list << "{A_V " << A_V_depth << "}\n";
			total_list << "{B_V " << B_V_depth << "}\n";
			total_list << "{C_V " << C_V_depth << "}\n";
			return total_list.str();
		}

		void set_num (int num , int* class_num) {
			(*class_num) = (*class_num) > num ? (*class_num) : num;
		}
	public:
		int gmem_A_depth;
		int gmem_B_depth;
		int gmem_C_depth;
		int A_V_depth;
		int B_V_depth;
		int C_V_depth;
		int trans_num;

	private:
		ofstream mFile;
		const char* mName;
};

extern "C" void kernel0 (
ap_uint<128>* A,
ap_uint<128>* B,
ap_uint<64>* C);

extern "C" void AESL_WRAP_kernel0 (
ap_uint<128>* A,
ap_uint<128>* B,
ap_uint<64>* C)
{
	refine_signal_handler();
	fstream wrapc_switch_file_token;
	wrapc_switch_file_token.open(".hls_cosim_wrapc_switch.log");
	int AESL_i;
	if (wrapc_switch_file_token.good())
	{
		CodeState = ENTER_WRAPC_PC;
		static unsigned AESL_transaction_pc = 0;
		string AESL_token;
		string AESL_num;
		static AESL_FILE_HANDLER aesl_fh;


		// output port post check: "gmem_C"
		aesl_fh.read(AUTOTB_TVOUT_PC_gmem_C, AESL_token); // [[transaction]]
		if (AESL_token != "[[transaction]]")
		{
			exit(1);
		}
		aesl_fh.read(AUTOTB_TVOUT_PC_gmem_C, AESL_num); // transaction number

		if (atoi(AESL_num.c_str()) == AESL_transaction_pc)
		{
			aesl_fh.read(AUTOTB_TVOUT_PC_gmem_C, AESL_token); // data

			sc_bv<64> *gmem_C_pc_buffer = new sc_bv<64>[32];
			int i = 0;

			while (AESL_token != "[[/transaction]]")
			{
				bool no_x = false;
				bool err = false;

				// search and replace 'X' with "0" from the 1st char of token
				while (!no_x)
				{
					size_t x_found = AESL_token.find('X');
					if (x_found != string::npos)
					{
						if (!err)
						{
							cerr << "WARNING: [SIM 212-201] RTL produces unknown value 'X' on port 'gmem_C', possible cause: There are uninitialized variables in the C design." << endl;
							err = true;
						}
						AESL_token.replace(x_found, 1, "0");
					}
					else
					{
						no_x = true;
					}
				}

				no_x = false;

				// search and replace 'x' with "0" from the 3rd char of token
				while (!no_x)
				{
					size_t x_found = AESL_token.find('x', 2);

					if (x_found != string::npos)
					{
						if (!err)
						{
							cerr << "WARNING: [SIM 212-201] RTL produces unknown value 'X' on port 'gmem_C', possible cause: There are uninitialized variables in the C design." << endl;
							err = true;
						}
						AESL_token.replace(x_found, 1, "0");
					}
					else
					{
						no_x = true;
					}
				}

				// push token into output port buffer
				if (AESL_token != "")
				{
					gmem_C_pc_buffer[i] = AESL_token.c_str();
					i++;
				}

				aesl_fh.read(AUTOTB_TVOUT_PC_gmem_C, AESL_token); // data or [[/transaction]]

				if (AESL_token == "[[[/runtime]]]" || aesl_fh.eof(AUTOTB_TVOUT_PC_gmem_C))
				{
					exit(1);
				}
			}

			// ***********************************
			if (i > 0)
			{
				// RTL Name: gmem_C
				{
					// bitslice(63, 0)
					// {
						// celement: C.V(63, 0)
						// {
							sc_lv<64>* C_V_lv0_0_31_1 = new sc_lv<64>[32];
						// }
					// }

					// bitslice(63, 0)
					{
						int hls_map_index = 0;
						// celement: C.V(63, 0)
						{
							// carray: (0) => (31) @ (1)
							for (int i_0 = 0; i_0 <= 31; i_0 += 1)
							{
								if (&(C[0]) != NULL) // check the null address if the c port is array or others
								{
									C_V_lv0_0_31_1[hls_map_index].range(63, 0) = sc_bv<64>(gmem_C_pc_buffer[hls_map_index].range(63, 0));
									hls_map_index++;
								}
							}
						}
					}

					// bitslice(63, 0)
					{
						int hls_map_index = 0;
						// celement: C.V(63, 0)
						{
							// carray: (0) => (31) @ (1)
							for (int i_0 = 0; i_0 <= 31; i_0 += 1)
							{
								// sub                    : i_0
								// ori_name               : C[i_0]
								// sub_1st_elem           : 0
								// ori_name_1st_elem      : C[0]
								// output_left_conversion : C[i_0]
								// output_type_conversion : (C_V_lv0_0_31_1[hls_map_index]).to_string(SC_BIN).c_str()
								if (&(C[0]) != NULL) // check the null address if the c port is array or others
								{
									C[i_0] = (C_V_lv0_0_31_1[hls_map_index]).to_string(SC_BIN).c_str();
									hls_map_index++;
								}
							}
						}
					}
				}
			}

			// release memory allocation
			delete [] gmem_C_pc_buffer;
		}

		AESL_transaction_pc++;
	}
	else
	{
		CodeState = ENTER_WRAPC;
		static unsigned AESL_transaction;

		static AESL_FILE_HANDLER aesl_fh;

		// "gmem_A"
		char* tvin_gmem_A = new char[50];
		aesl_fh.touch(AUTOTB_TVIN_gmem_A);

		// "gmem_B"
		char* tvin_gmem_B = new char[50];
		aesl_fh.touch(AUTOTB_TVIN_gmem_B);

		// "gmem_C"
		char* tvin_gmem_C = new char[50];
		aesl_fh.touch(AUTOTB_TVIN_gmem_C);
		char* tvout_gmem_C = new char[50];
		aesl_fh.touch(AUTOTB_TVOUT_gmem_C);

		// "A_V"
		char* tvin_A_V = new char[50];
		aesl_fh.touch(AUTOTB_TVIN_A_V);

		// "B_V"
		char* tvin_B_V = new char[50];
		aesl_fh.touch(AUTOTB_TVIN_B_V);

		// "C_V"
		char* tvin_C_V = new char[50];
		aesl_fh.touch(AUTOTB_TVIN_C_V);

		CodeState = DUMP_INPUTS;
		static INTER_TCL_FILE tcl_file(INTER_TCL);
		int leading_zero;

		// [[transaction]]
		sprintf(tvin_gmem_A, "[[transaction]] %d\n", AESL_transaction);
		aesl_fh.write(AUTOTB_TVIN_gmem_A, tvin_gmem_A);

		sc_bv<128>* gmem_A_tvin_wrapc_buffer = new sc_bv<128>[16];

		// RTL Name: gmem_A
		{
			// bitslice(127, 0)
			{
				int hls_map_index = 0;
				// celement: A.V(127, 0)
				{
					// carray: (0) => (15) @ (1)
					for (int i_0 = 0; i_0 <= 15; i_0 += 1)
					{
						// sub                   : i_0
						// ori_name              : A[i_0]
						// sub_1st_elem          : 0
						// ori_name_1st_elem     : A[0]
						// regulate_c_name       : A_V
						// input_type_conversion : (A[i_0]).to_string(2).c_str()
						if (&(A[0]) != NULL) // check the null address if the c port is array or others
						{
							sc_lv<128> A_V_tmp_mem;
							A_V_tmp_mem = (A[i_0]).to_string(2).c_str();
							gmem_A_tvin_wrapc_buffer[hls_map_index].range(127, 0) = A_V_tmp_mem.range(127, 0);
                                 	       hls_map_index++;
						}
					}
				}
			}
		}

		// dump tv to file
		for (int i = 0; i < 16; i++)
		{
			sprintf(tvin_gmem_A, "%s\n", (gmem_A_tvin_wrapc_buffer[i]).to_string(SC_HEX).c_str());
			aesl_fh.write(AUTOTB_TVIN_gmem_A, tvin_gmem_A);
		}

		tcl_file.set_num(16, &tcl_file.gmem_A_depth);
		sprintf(tvin_gmem_A, "[[/transaction]] \n");
		aesl_fh.write(AUTOTB_TVIN_gmem_A, tvin_gmem_A);

		// release memory allocation
		delete [] gmem_A_tvin_wrapc_buffer;

		// [[transaction]]
		sprintf(tvin_gmem_B, "[[transaction]] %d\n", AESL_transaction);
		aesl_fh.write(AUTOTB_TVIN_gmem_B, tvin_gmem_B);

		sc_bv<128>* gmem_B_tvin_wrapc_buffer = new sc_bv<128>[16];

		// RTL Name: gmem_B
		{
			// bitslice(127, 0)
			{
				int hls_map_index = 0;
				// celement: B.V(127, 0)
				{
					// carray: (0) => (15) @ (1)
					for (int i_0 = 0; i_0 <= 15; i_0 += 1)
					{
						// sub                   : i_0
						// ori_name              : B[i_0]
						// sub_1st_elem          : 0
						// ori_name_1st_elem     : B[0]
						// regulate_c_name       : B_V
						// input_type_conversion : (B[i_0]).to_string(2).c_str()
						if (&(B[0]) != NULL) // check the null address if the c port is array or others
						{
							sc_lv<128> B_V_tmp_mem;
							B_V_tmp_mem = (B[i_0]).to_string(2).c_str();
							gmem_B_tvin_wrapc_buffer[hls_map_index].range(127, 0) = B_V_tmp_mem.range(127, 0);
                                 	       hls_map_index++;
						}
					}
				}
			}
		}

		// dump tv to file
		for (int i = 0; i < 16; i++)
		{
			sprintf(tvin_gmem_B, "%s\n", (gmem_B_tvin_wrapc_buffer[i]).to_string(SC_HEX).c_str());
			aesl_fh.write(AUTOTB_TVIN_gmem_B, tvin_gmem_B);
		}

		tcl_file.set_num(16, &tcl_file.gmem_B_depth);
		sprintf(tvin_gmem_B, "[[/transaction]] \n");
		aesl_fh.write(AUTOTB_TVIN_gmem_B, tvin_gmem_B);

		// release memory allocation
		delete [] gmem_B_tvin_wrapc_buffer;

		// [[transaction]]
		sprintf(tvin_gmem_C, "[[transaction]] %d\n", AESL_transaction);
		aesl_fh.write(AUTOTB_TVIN_gmem_C, tvin_gmem_C);

		sc_bv<64>* gmem_C_tvin_wrapc_buffer = new sc_bv<64>[32];

		// RTL Name: gmem_C
		{
			// bitslice(63, 0)
			{
				int hls_map_index = 0;
				// celement: C.V(63, 0)
				{
					// carray: (0) => (31) @ (1)
					for (int i_0 = 0; i_0 <= 31; i_0 += 1)
					{
						// sub                   : i_0
						// ori_name              : C[i_0]
						// sub_1st_elem          : 0
						// ori_name_1st_elem     : C[0]
						// regulate_c_name       : C_V
						// input_type_conversion : (C[i_0]).to_string(2).c_str()
						if (&(C[0]) != NULL) // check the null address if the c port is array or others
						{
							sc_lv<64> C_V_tmp_mem;
							C_V_tmp_mem = (C[i_0]).to_string(2).c_str();
							gmem_C_tvin_wrapc_buffer[hls_map_index].range(63, 0) = C_V_tmp_mem.range(63, 0);
                                 	       hls_map_index++;
						}
					}
				}
			}
		}

		// dump tv to file
		for (int i = 0; i < 32; i++)
		{
			sprintf(tvin_gmem_C, "%s\n", (gmem_C_tvin_wrapc_buffer[i]).to_string(SC_HEX).c_str());
			aesl_fh.write(AUTOTB_TVIN_gmem_C, tvin_gmem_C);
		}

		tcl_file.set_num(32, &tcl_file.gmem_C_depth);
		sprintf(tvin_gmem_C, "[[/transaction]] \n");
		aesl_fh.write(AUTOTB_TVIN_gmem_C, tvin_gmem_C);

		// release memory allocation
		delete [] gmem_C_tvin_wrapc_buffer;

		// [[transaction]]
		sprintf(tvin_A_V, "[[transaction]] %d\n", AESL_transaction);
		aesl_fh.write(AUTOTB_TVIN_A_V, tvin_A_V);

		sc_bv<32> A_V_tvin_wrapc_buffer;

		// RTL Name: A_V
		{
		}

		// dump tv to file
		for (int i = 0; i < 1; i++)
		{
			sprintf(tvin_A_V, "%s\n", (A_V_tvin_wrapc_buffer).to_string(SC_HEX).c_str());
			aesl_fh.write(AUTOTB_TVIN_A_V, tvin_A_V);
		}

		tcl_file.set_num(1, &tcl_file.A_V_depth);
		sprintf(tvin_A_V, "[[/transaction]] \n");
		aesl_fh.write(AUTOTB_TVIN_A_V, tvin_A_V);

		// [[transaction]]
		sprintf(tvin_B_V, "[[transaction]] %d\n", AESL_transaction);
		aesl_fh.write(AUTOTB_TVIN_B_V, tvin_B_V);

		sc_bv<32> B_V_tvin_wrapc_buffer;

		// RTL Name: B_V
		{
		}

		// dump tv to file
		for (int i = 0; i < 1; i++)
		{
			sprintf(tvin_B_V, "%s\n", (B_V_tvin_wrapc_buffer).to_string(SC_HEX).c_str());
			aesl_fh.write(AUTOTB_TVIN_B_V, tvin_B_V);
		}

		tcl_file.set_num(1, &tcl_file.B_V_depth);
		sprintf(tvin_B_V, "[[/transaction]] \n");
		aesl_fh.write(AUTOTB_TVIN_B_V, tvin_B_V);

		// [[transaction]]
		sprintf(tvin_C_V, "[[transaction]] %d\n", AESL_transaction);
		aesl_fh.write(AUTOTB_TVIN_C_V, tvin_C_V);

		sc_bv<32> C_V_tvin_wrapc_buffer;

		// RTL Name: C_V
		{
		}

		// dump tv to file
		for (int i = 0; i < 1; i++)
		{
			sprintf(tvin_C_V, "%s\n", (C_V_tvin_wrapc_buffer).to_string(SC_HEX).c_str());
			aesl_fh.write(AUTOTB_TVIN_C_V, tvin_C_V);
		}

		tcl_file.set_num(1, &tcl_file.C_V_depth);
		sprintf(tvin_C_V, "[[/transaction]] \n");
		aesl_fh.write(AUTOTB_TVIN_C_V, tvin_C_V);

// [call_c_dut] ---------->

		CodeState = CALL_C_DUT;
		kernel0(A, B, C);

		CodeState = DUMP_OUTPUTS;

		// [[transaction]]
		sprintf(tvout_gmem_C, "[[transaction]] %d\n", AESL_transaction);
		aesl_fh.write(AUTOTB_TVOUT_gmem_C, tvout_gmem_C);

		sc_bv<64>* gmem_C_tvout_wrapc_buffer = new sc_bv<64>[32];

		// RTL Name: gmem_C
		{
			// bitslice(63, 0)
			{
				int hls_map_index = 0;
				// celement: C.V(63, 0)
				{
					// carray: (0) => (31) @ (1)
					for (int i_0 = 0; i_0 <= 31; i_0 += 1)
					{
						// sub                   : i_0
						// ori_name              : C[i_0]
						// sub_1st_elem          : 0
						// ori_name_1st_elem     : C[0]
						// regulate_c_name       : C_V
						// input_type_conversion : (C[i_0]).to_string(2).c_str()
						if (&(C[0]) != NULL) // check the null address if the c port is array or others
						{
							sc_lv<64> C_V_tmp_mem;
							C_V_tmp_mem = (C[i_0]).to_string(2).c_str();
							gmem_C_tvout_wrapc_buffer[hls_map_index].range(63, 0) = C_V_tmp_mem.range(63, 0);
                                 	       hls_map_index++;
						}
					}
				}
			}
		}

		// dump tv to file
		for (int i = 0; i < 32; i++)
		{
			sprintf(tvout_gmem_C, "%s\n", (gmem_C_tvout_wrapc_buffer[i]).to_string(SC_HEX).c_str());
			aesl_fh.write(AUTOTB_TVOUT_gmem_C, tvout_gmem_C);
		}

		tcl_file.set_num(32, &tcl_file.gmem_C_depth);
		sprintf(tvout_gmem_C, "[[/transaction]] \n");
		aesl_fh.write(AUTOTB_TVOUT_gmem_C, tvout_gmem_C);

		// release memory allocation
		delete [] gmem_C_tvout_wrapc_buffer;

		CodeState = DELETE_CHAR_BUFFERS;
		// release memory allocation: "gmem_A"
		delete [] tvin_gmem_A;
		// release memory allocation: "gmem_B"
		delete [] tvin_gmem_B;
		// release memory allocation: "gmem_C"
		delete [] tvout_gmem_C;
		delete [] tvin_gmem_C;
		// release memory allocation: "A_V"
		delete [] tvin_A_V;
		// release memory allocation: "B_V"
		delete [] tvin_B_V;
		// release memory allocation: "C_V"
		delete [] tvin_C_V;

		AESL_transaction++;

		tcl_file.set_num(AESL_transaction , &tcl_file.trans_num);
	}
}

