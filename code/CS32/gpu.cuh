/*
GNU GENERAL PUBLIC LICENSE - Version 3

Copyright (c) 2018 Vitor Curtis <vitorcurtis@gmail.com>, Carlos Sanches <alonso@ita.br>
*/

#pragma once

#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include "main.h"

#ifndef GPU_CUH
#define GPU_CUH

#ifdef _WIN32
#define MAX_THREADS		768
#define MAX_BLOCKS		4
#else
#define MAX_THREADS		768
#define MAX_BLOCKS		14
#endif

#define CUDAErrorExit(funcName) if (cudaStatus != cudaSuccess) {printError(funcName);exit(-1);}
#define CUDAErrorGoto(funcName) if (cudaStatus != cudaSuccess) {printError(funcName);goto Error;}

#endif // GPU_CUH
