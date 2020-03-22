// ==============================================================
// Vivado(TM) HLS - High-Level Synthesis from C, C++ and SystemC v2019.2 (64-bit)
// Copyright 1986-2019 Xilinx, Inc. All Rights Reserved.
// ==============================================================
#ifndef XKERNEL0_H
#define XKERNEL0_H

#ifdef __cplusplus
extern "C" {
#endif

/***************************** Include Files *********************************/
#ifndef __linux__
#include "xil_types.h"
#include "xil_assert.h"
#include "xstatus.h"
#include "xil_io.h"
#else
#include <stdint.h>
#include <assert.h>
#include <dirent.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <stddef.h>
#endif
#include "xkernel0_hw.h"

/**************************** Type Definitions ******************************/
#ifdef __linux__
typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
#else
typedef struct {
    u16 DeviceId;
    u32 Control_BaseAddress;
} XKernel0_Config;
#endif

typedef struct {
    u32 Control_BaseAddress;
    u32 IsReady;
} XKernel0;

/***************** Macros (Inline Functions) Definitions *********************/
#ifndef __linux__
#define XKernel0_WriteReg(BaseAddress, RegOffset, Data) \
    Xil_Out32((BaseAddress) + (RegOffset), (u32)(Data))
#define XKernel0_ReadReg(BaseAddress, RegOffset) \
    Xil_In32((BaseAddress) + (RegOffset))
#else
#define XKernel0_WriteReg(BaseAddress, RegOffset, Data) \
    *(volatile u32*)((BaseAddress) + (RegOffset)) = (u32)(Data)
#define XKernel0_ReadReg(BaseAddress, RegOffset) \
    *(volatile u32*)((BaseAddress) + (RegOffset))

#define Xil_AssertVoid(expr)    assert(expr)
#define Xil_AssertNonvoid(expr) assert(expr)

#define XST_SUCCESS             0
#define XST_DEVICE_NOT_FOUND    2
#define XST_OPEN_DEVICE_FAILED  3
#define XIL_COMPONENT_IS_READY  1
#endif

/************************** Function Prototypes *****************************/
#ifndef __linux__
int XKernel0_Initialize(XKernel0 *InstancePtr, u16 DeviceId);
XKernel0_Config* XKernel0_LookupConfig(u16 DeviceId);
int XKernel0_CfgInitialize(XKernel0 *InstancePtr, XKernel0_Config *ConfigPtr);
#else
int XKernel0_Initialize(XKernel0 *InstancePtr, const char* InstanceName);
int XKernel0_Release(XKernel0 *InstancePtr);
#endif

void XKernel0_Start(XKernel0 *InstancePtr);
u32 XKernel0_IsDone(XKernel0 *InstancePtr);
u32 XKernel0_IsIdle(XKernel0 *InstancePtr);
u32 XKernel0_IsReady(XKernel0 *InstancePtr);
void XKernel0_EnableAutoRestart(XKernel0 *InstancePtr);
void XKernel0_DisableAutoRestart(XKernel0 *InstancePtr);

void XKernel0_Set_A_V(XKernel0 *InstancePtr, u32 Data);
u32 XKernel0_Get_A_V(XKernel0 *InstancePtr);
void XKernel0_Set_B_V(XKernel0 *InstancePtr, u32 Data);
u32 XKernel0_Get_B_V(XKernel0 *InstancePtr);
void XKernel0_Set_C_V(XKernel0 *InstancePtr, u32 Data);
u32 XKernel0_Get_C_V(XKernel0 *InstancePtr);

void XKernel0_InterruptGlobalEnable(XKernel0 *InstancePtr);
void XKernel0_InterruptGlobalDisable(XKernel0 *InstancePtr);
void XKernel0_InterruptEnable(XKernel0 *InstancePtr, u32 Mask);
void XKernel0_InterruptDisable(XKernel0 *InstancePtr, u32 Mask);
void XKernel0_InterruptClear(XKernel0 *InstancePtr, u32 Mask);
u32 XKernel0_InterruptGetEnabled(XKernel0 *InstancePtr);
u32 XKernel0_InterruptGetStatus(XKernel0 *InstancePtr);

#ifdef __cplusplus
}
#endif

#endif
