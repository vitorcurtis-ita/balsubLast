/*
GNU GENERAL PUBLIC LICENSE - Version 3

Copyright (c) 2018 Vitor Curtis <vitorcurtis@gmail.com>, Carlos Sanches <alonso@ita.br>
*/

#include "gpu.cuh"
#include "Alg7.c"
#include "Alg8.c"
#include "Kernels.cu"

int blocksPerGrid, threadsPerBlock;
double cStart, cEnd, c1, c2, cToHost;



// Helper function for using CUDA kernel
cudaError_t Alg2() {
	cudaError_t cudaStatus = cudaSuccess;
	
	_N_TYPE i;				
	// Indexes
	_C_TYPE b, e;							// Indexes
	_N_TYPE* gm=NULL;						// G vector for GPU
	_ARCH_TYPE* gb=NULL;					// Binary vector for GPU
	_ARCH_TYPE* u=NULL;						// Update vector for GPU
	

	
	// If a solution was found, exit
	if (Alg7()) return cudaStatus;

	// Allocating the bit G vector for GPU
	cudaStatus = cudaMalloc((void**)&gb, sizeof(_ARCH_TYPE)*(c-Wmin+1+ARCH-1)/ARCH );
	CUDAErrorGoto("cudaMalloc");
	cudaMemsetAsync(gb, 0, sizeof(_ARCH_TYPE)*(c-Wmin+1+ARCH-1)/ARCH );
	CUDAErrorGoto("cudaMemset");
	cudaStatus = cudaMalloc((void**)&u, sizeof(_ARCH_TYPE)*(c-Wmin+1+ARCH-1)/ARCH );
	CUDAErrorGoto("cudaMalloc");
	cudaMemsetAsync(u, 0, sizeof(_ARCH_TYPE)*(c-Wmin+1+ARCH-1)/ARCH );
	CUDAErrorGoto("cudaMemset");

	// Register the G vector for GPU
	printf("Alocando fisicamente %llu bytes, c=%llu, Wmin=%llu\n", (unsigned long long)(c-Wmin+1)*sizeof(_N_TYPE), (unsigned long long)c, (unsigned long long)Wmin);
	cudaStatus = cudaHostRegister(&G[Wmin], (c-Wmin+1)*sizeof(_N_TYPE), cudaHostRegisterMapped);
	CUDAErrorGoto("cudaHostRegister");
	cudaStatus = cudaHostGetDevicePointer((void**)&gm, &G[Wmin], 0);
	gm = &gm[-((int)Wmin)];		// Correction
	


	// Ajusting variables to calculate b and e
	sum1_i = W[1];
	sumi1_n = sumW - W[1];

	// Setting the first item
	if (W[1] <= c) {
		KSetFirst<<<1,1>>>(Wmin, W[1], gb);
		G[W[1]] = 1;
	}



	//cudaStreamSynchronize(0);
	//c1 = wall_time();

	// Dynamic Program - For all next integers
	for(i=2; i<=n; i++) {
		sumi1_n -= W[i];
		sum1_i += W[i];

		// c must be smaller or equal
		if (W[i] > c) continue;

		// Optimization
		b = (t>sumi1_n && t-sumi1_n>(_C_TYPE)W[i]+(_C_TYPE)W[i-1]) ? t-sumi1_n : W[i]+W[i-1];
		e = min(c, sum1_i);

		blocksPerGrid = ((e-b+1)+MAX_THREADS-1)/MAX_THREADS;
		blocksPerGrid = blocksPerGrid>MAX_BLOCKS ? MAX_BLOCKS : blocksPerGrid;
		threadsPerBlock = MAX_THREADS;

		if (b <= e) {
			K1<<<blocksPerGrid, threadsPerBlock>>>(Wmin, c, i, b, e, W[i], gm, gb, u);
			// Alignment for b, including misaligned capacities
			// Setting Wi
			K2<<<blocksPerGrid, threadsPerBlock>>>(Wmin, c, i, b-mod_q(b-Wmin), e, W[i], gm, gb, u);
		} else {
			// Alignment for b, including misaligned capacities
			// Setting Wi
			K2<<<1, threadsPerBlock>>>(Wmin, c, i, b-mod_q(b-Wmin), e, W[i], gm, gb, u);
		}
	}



	// Timing kernel
	CUDAErrorGoto(cudaGetErrorString(cudaStatus));
	cudaStatus = cudaDeviceSynchronize();
	CUDAErrorGoto("cudaDeviceSynchronize");
	//c2 = wall_time();

	// Unregister the memory
	cudaStatus = cudaHostUnregister(&G[Wmin]);
	CUDAErrorGoto("cudaHostUnregister");
	//cToHost = wall_time();

	
	// Retrieving the solution
	// Best solution found (Not applicable in parallel version)
	for (s=c; G[s]==n+1; s--);
	
	Alg8();

Error:
	if (gm != NULL) cudaFree(gm);
	if (u != NULL) cudaFree(u);
	if (&G[Wmin] != NULL) free(&G[Wmin]);

    return cudaStatus;
}

int main(int argc, char *argv[]) {
	cudaError_t cudaStatus;
	_N_TYPE i;
	_C_TYPE aux;

	// Reading file
	init(argc, &argv[0]);
	fprintf(stdout, "File readed\n");
	fprintf(stdout, "\nSummary K1:\n");
	fprintf(stdout, "Problem with n=%llu, t=%llu\n", (unsigned long long)n, (unsigned long long)targetSum);
	fflush(stdout);

	// Checking instance
	aux = 0;
	for(i=1; i<=n; i++) {
		if (W[i] >= targetSum) {
			fprintf(stderr, "Error: trivial instance where capacity is less or equal to the %d-th item!\n", i);
			exit(-1);
		}

		aux += W[i];
	}
	if (targetSum >= aux) {
		fprintf(stderr, "Error: trivial instance where capacity is equal or greater than the sum of all integers!\n");
		exit(-1);
	}

	// Set Map Pinned Mem
	cudaStatus = cudaSetDeviceFlags(cudaDeviceMapHost);
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Error: cudaSetDeviceFlags failed!  Do you have a CUDA-capable GPU installed?\n");
        return -1;
    }



	cStart = wall_time();
	
	cudaStatus = Alg2();
	CUDAErrorExit("Kernel call");

	cEnd = wall_time();
	
	
	
	// cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Parallel Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
	CUDAErrorExit("cudaDeviceReset");

	// Checking the solution
	for(i=1,aux=0; i<=n; i++)
		if (X[i]) aux += W[i];

	fprintf(stdout, "Solution: %llu=%llu for capacity %llu\n\n", (unsigned long long)s, (unsigned long long)aux, (unsigned long long)targetSum);
	fprintf(stdout, "Ticks: \t\t%lf seg\n", (cEnd-cStart));
	
	// Alg, n, solution, capacity, seg
	fprintf(stderr, "CSp,n=%llu,s=%llu,c=%llu,time=%lf\n", (unsigned long long)n, (unsigned long long)s, (unsigned long long)targetSum, (cEnd-cStart));

	exit(0);
}
