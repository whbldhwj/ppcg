// ==============================================================
// Vivado(TM) HLS - High-Level Synthesis from C, C++ and SystemC v2019.2 (64-bit)
// Copyright 1986-2019 Xilinx, Inc. All Rights Reserved.
// ==============================================================
#ifndef __linux__

#include "xstatus.h"
#include "xparameters.h"
#include "xkernel0.h"

extern XKernel0_Config XKernel0_ConfigTable[];

XKernel0_Config *XKernel0_LookupConfig(u16 DeviceId) {
	XKernel0_Config *ConfigPtr = NULL;

	int Index;

	for (Index = 0; Index < XPAR_XKERNEL0_NUM_INSTANCES; Index++) {
		if (XKernel0_ConfigTable[Index].DeviceId == DeviceId) {
			ConfigPtr = &XKernel0_ConfigTable[Index];
			break;
		}
	}

	return ConfigPtr;
}

int XKernel0_Initialize(XKernel0 *InstancePtr, u16 DeviceId) {
	XKernel0_Config *ConfigPtr;

	Xil_AssertNonvoid(InstancePtr != NULL);

	ConfigPtr = XKernel0_LookupConfig(DeviceId);
	if (ConfigPtr == NULL) {
		InstancePtr->IsReady = 0;
		return (XST_DEVICE_NOT_FOUND);
	}

	return XKernel0_CfgInitialize(InstancePtr, ConfigPtr);
}

#endif

