/*
GNU GENERAL PUBLIC LICENSE - Version 3

Copyright (c) 2018 Vitor Curtis <vitorcurtis@gmail.com>, Carlos Sanches <alonso@ita.br>
*/

#pragma once

#include "gpu.cuh"
#include "main.h"

#ifndef KERNELS_C
#define KERNELS_C

__global__ void K1(const _WI_TYPE Wmin, const _C_TYPE c, const _N_TYPE i, const _C_TYPE b, const _C_TYPE e, const _WI_TYPE Wi, _N_TYPE* __restrict__ gm, _ARCH_TYPE* __restrict__ gb, _ARCH_TYPE* __restrict__ u)
{
	// Calculating indexes
	const unsigned int inc = (gridDim.x*blockDim.x)*ARCH;
	const unsigned int tid = (blockDim.x*blockIdx.x + threadIdx.x)*ARCH;
	
	_C_TYPE j, k;
	_ARCH_TYPE cap, comp, right, left;
	
	// Solution found
	//if (G[c]!=n+1) return;	// Not included in pseudocode



	// Alignment for cap
	j = b+ ARCH-1 - mod_q(b+ ARCH-1 -Wmin);
	j += tid;

	// Ignored misaligned capacities, starting from b
	if (tid==0 && b!=j) {
		cap = gb[(b-Wmin)/ARCH];

		k = b -Wi -Wmin;
		right = gb[k/ARCH] >> mod_q(k);
		left = gb[(k+ARCH-1)/ARCH] << (ARCH-mod_q(k));
		comp = left|right;

		// Alignment with cap from b, with 0 in begining
		comp = comp << mod_q(b-Wmin);

		// Condition to write
		comp = comp&(~cap);
		if (comp) {
			u[(b-Wmin)/ARCH] = cap | comp;						// Update

			// Back b to bit0
			comp = comp >> mod_q(b-Wmin);

			// Set CPU memory
			for(_ARCH_TYPE bit=0; comp; bit++, comp=comp>>1)
				if (comp&1 && b+bit<=c) {
					gm[b+bit]=i;
				}
		}
	}

	// Computing the matrix S
	for (; j<=e; j+=inc) {
		cap = gb[(j-Wmin)/ARCH];

		k = j -Wi -Wmin;
		right = gb[k/ARCH] >> mod_q(k);
		left = gb[(k+ARCH-1)/ARCH] << (ARCH-mod_q(k));
		comp = left|right;

		// Condition to write
		comp = comp&(~cap);
		if (comp) {
			u[(j-Wmin)/ARCH] = cap | comp;		// Update

			// Set CPU memory
			for(_ARCH_TYPE bit=0; comp; bit++, comp=comp>>1)
				if (comp&1 && j+bit<=c) {
					gm[j+bit]=i;
				}
		}
	}
}

__global__ void K2(const _WI_TYPE Wmin, const _C_TYPE c, const _N_TYPE i, const _C_TYPE b, const _C_TYPE e, const _WI_TYPE Wi, _N_TYPE* __restrict__ gm, _ARCH_TYPE* __restrict__ gb, _ARCH_TYPE* __restrict__ u)
{
	// Calculating indexes
	const unsigned int inc = (gridDim.x*blockDim.x)*ARCH;
	const unsigned int tid = (blockDim.x*blockIdx.x + threadIdx.x)*ARCH;
	
	_C_TYPE j;

	// Solution found
	//if (G[c]!=n+1) return;	// Not included in pseudocode
	
	// Set Wi
	const _WI_TYPE Wi_ = Wi - Wmin;
	if (tid==0 && (gb[Wi_/ARCH] & (1<<mod_q(Wi_))) == 0) {
		gb[Wi_/ARCH] |= (1<<mod_q(Wi_));
		gm[Wi] = i;
	}

	// For all capacities
	for (j=b+tid-Wmin; j<=e-Wmin; j+=inc) {				// for (j=b+tid; j<=e; j+=inc) {
		if (u[j/ARCH]) {								//   if (u[(j-Wmin)/Q]) {
			gb[j/ARCH] = u[j/ARCH];						//       gb[(j-Wmin)/Q] = u[(j-Wmin)/Q];
			u[j/ARCH] = 0;								//       u[(j-Wmin)/Q] = 0;
		}												//   }
	}													// }
}

__global__ void KSetFirst(const _WI_TYPE Wmin, const _WI_TYPE Wi, _ARCH_TYPE* __restrict__ gb)
{
	// Set W[1]
	gb[(Wi-Wmin)/ARCH] = 1<<(mod_q(Wi-Wmin));
}

#endif // KERNELS_C