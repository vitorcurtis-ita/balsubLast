/*
GNU GENERAL PUBLIC LICENSE - Version 3

Copyright (c) 2018 Vitor Curtis <vitorcurtis@gmail.com>, Carlos Sanches <alonso@ita.br>
*/

/**
 * Primeira versão funcional
 */

#include "main.h"

double cStart, cEnd, c1, c2, cToHost;

unsigned char* X;						// Solution vector
_WI_TYPE* W;							// List of integers

_C_TYPE targetSum;						// Original sum to search
_C_TYPE c;								// Sum to search
_C_TYPE solutionFound;					// Best solution known
_N_TYPE n;								// Amount of integers in the List

_WI_TYPE Wmax;							// lightest and heaviest integers in W

unsigned int tries;						// times it have to rerun the core algoritm



STIN void initVectors(const _WI_TYPE kTableIdx_c, const _WI_TYPE kTableIdx_max, _N_TYPE* RV, _N_TYPE* AV, _N_TYPE* T) {
	_C_TYPE k;

	for (k = 0; k <= kTableIdx_c; k++) RV[k] = AV[k] = 0;											// k -> [c-(Wmax-1), c]
	for (k = kTableIdx_c + 1; k <= kTableIdx_max; k++) { RV[k] = T[k] = 1; AV[k] = 0; }				// k -> [c+1, c+(Wmax-1)]
}



STIN void ARcomp(const _WI_TYPE k_break, const _WI_TYPE kTableIdx_c, const _N_TYPE b, const _WI_TYPE nextA, const _WI_TYPE nextR, _N_TYPE* RV, _N_TYPE* AV, _N_TYPE* T) {
	_N_TYPE i;
	_C_TYPE k;

	for (i = b; i <= nextA; i++) {

		// Inserting the i-th item
		_C_TYPE maxFound = 0;							// Maximum combination found durring inserting

		for (k = kTableIdx_c - 1; k < kTableIdx_c; k--) {				// k -> [c-(Wmax-1), c[
			_C_TYPE newK = k + W[i];					// New: atual capacity + i-th item

			if (RV[k] > RV[newK]) {						// Keep the combination with most items previously inserted (closer to b)
														// Remove items from RV[k]-1 to RV[newK]
				if (newK > kTableIdx_c)					// Temporary buffer which checks the limit for remove step
					T[newK] = RV[newK];
				RV[newK] = RV[k];						// Needed?
				AV[newK] = i;

				if (newK == k_break &&					// Break if found solution (Prevent overwrite)
					AV[newK] <= nextA &&
					RV[newK] >= nextR
					) return;

				if (maxFound == 0)
					maxFound = newK;
			}
		}

		// Removing items to find new combinations
		for (k = maxFound; k > kTableIdx_c; k--) {		// k -> ]c, c+(Wmax-1)]
														// Remove all items previously not tested
			_N_TYPE limite = (T[k]>nextR) ? T[k] : nextR;
			for (_N_TYPE removeI = RV[k] - 1; removeI >= limite; removeI--) {	// All inserted items b(-1) to ... previous
				_C_TYPE newK = k - W[removeI];

				if (removeI > RV[newK]) {
					RV[newK] = removeI;
					AV[newK] = AV[k];					// Needed?

					if (newK == k_break &&				// Break if found solution (Prevent overwrite)
						AV[newK] <= nextA &&
						RV[newK] >= nextR
						) return;
				}

			}
			T[k] = RV[k];					// Next time, remove up to RV[k]
		}
	}
}



STIN _C_TYPE BalsubLast(_C_TYPE c) {
	_N_TYPE b;								// Break item
	_C_TYPE bSum;							// Break solution (>=c): sum up to break item (excluded)
	_N_TYPE i;
	_WI_TYPE k;
	_WI_TYPE nextA, nextR;
	_C_TYPE solution;

	_N_TYPE* RV;							// Removed itens vector
	_N_TYPE* AV;							// Added itens vector
	_N_TYPE* T;								// Temporary vector



	tries = 0;
	
	// Calculating break item, break solution and Wmax
	bSum = 0;
	b = 1;
	for (i = 1; i <= n; i++) {
		if (bSum + W[i] <= c) {
			bSum += W[i];
			b++;									// break item overflows the c value
		}
	}
	Wmax = W[n];									// sorted in ascending order, so

	// Size of vectors:
	// Limits: [c+1-W[b-1]:c-1+Wmax] => ]c-W[b-1]:c+Wmax[
	// Size: c-1+Wmax - offset + 1 = c+Wmax-offset
	// Size of second part: ]c:c+Wmax[ => c-1+Wmax - (c+1) + 1 = Wmax-1

	const _C_TYPE offset = (b == 1) ? bSum : min(bSum, c + 1 - W[b - 1]);
	const _WI_TYPE kTableIdx_c = c-offset;
	const _WI_TYPE kTableIdx_max = kTableIdx_c-1+Wmax;

	// Allocating vectors
	AV = (_N_TYPE*)malloc((kTableIdx_max+1) * sizeof(_N_TYPE));
	RV = (_N_TYPE*)malloc((kTableIdx_max+1) * sizeof(_N_TYPE));
	T = (_N_TYPE*)malloc((Wmax-1) * sizeof(_N_TYPE));
	T = &T[-(long long)(kTableIdx_c + 1)];

	if (AV == NULL || RV == NULL || T == NULL) {
		fprintf(stderr, "Error: malloc(%ul) at %s, line %d!\n", (unsigned long)(kTableIdx_max+1) * sizeof(_N_TYPE), __FILE__, __LINE__);
		exit(-1);
	}

	// Start with break solution
	for (i = 1; i < b; i++)	X[i] = TRUE;
	for (; i <= n; i++)	X[i] = FALSE;



	// The first time, search for c
	k = kTableIdx_c;
	nextA = n;
	nextR = 1;

	// Init vectors and fill the break solution
	initVectors(kTableIdx_c, kTableIdx_max, RV, AV, T);
	RV[bSum - offset] = b;
	AV[bSum - offset] = b - 1;

	// Main loop
	ARcomp(k, kTableIdx_c, b, nextA, nextR, RV, AV, T);

	// Search for the best solution
	for (k = kTableIdx_c; RV[k] == 0; k--);
	solution = k + offset;



	// Recover the items
	while (k != bSum - offset) {		// Compute until it reaches the break solution

		// The last operations was REM RV[k] or ADD AV[k]

		// Check if: REM RV[k]: k' - W[REM] = k	->		k' = k + W[RV[k]]
		if (RV[k]) {	// Prevent overflow
			_N_TYPE r = RV[k];
			_WI_TYPE w = W[r];
			if (k + w <= kTableIdx_max && r < RV[k + w] && AV[k] >= AV[k + w]) {
				X[r] = 0;
				k = k + w;
				nextR = r+1;
				continue;
			}
		}

		// Check if: ADD AV[k]: k' + W[ADD] = k	->		k' = k - W[AV[k]]
		if (AV[k]) {
			_N_TYPE a = AV[k];
			_WI_TYPE w = W[a];
			if (k >= w && a > AV[k - w] && RV[k] <= RV[k - w]) {
				X[a] = 1;
				k = k - w;
				nextA = a-1;
				continue;
			}
		}

		// Not able to decide between ADD and REM, run again

		// Init vectors and fill the break solution
		initVectors(kTableIdx_c, kTableIdx_max, RV, AV, T);
		RV[bSum - offset] = b;
		AV[bSum - offset] = b - 1;

		// Main loop
		ARcomp(k, kTableIdx_c, b, nextA, nextR, RV, AV, T);
		tries++;
	}



	free(AV);
	free(RV);
	free(&T[kTableIdx_c + 1]);

	return solution;
}

STIN void checkInstance() {
	_N_TYPE i;
	_WI_TYPE tmp;
	_C_TYPE aux;

	aux = 0;
	for (i = 1; i <= n; i++) {
		if (W[i] == targetSum) {
			fprintf(stderr, "Error: trivial instance where capacity is equal to the %d-th item!\n", i);
			exit(-1);
		}

		aux += W[i];
	}
	if (targetSum >= aux) {
		fprintf(stderr, "Error: trivial instance where capacity is equal or greater than the sum of all integers!\n");
		exit(-1);
	}

	// Ignoring high weights
	for (i = 1; i <= n;) {
		if (W[i] > targetSum) {
			_swap(W[i], W[n]);
			n--;
		}
		else {
			i++;
		}
	}
	if (n <= 2) {
		fprintf(stderr, "Error: very small instance: n=%lu!\n", (unsigned long)n);
		exit(-1);
	}
}

int main(int argc, char *argv[]) {
	_N_TYPE i;
	_C_TYPE aux;

	// Reading file
	init(argc, &argv[0]);
	fprintf(stdout, "File readed\n");
	fprintf(stdout, "\nSummary BalAR:\n");
	fprintf(stdout, "Problem with n=%llu, t=%llu\n", (unsigned long long)n, (unsigned long long)targetSum);
	fflush(stdout);

	// Checking instance
	checkInstance();

	// Sorting items
	qsort(&W[1], n, sizeof(_WI_TYPE), compareAscending);



	cStart = wall_time();
	
	// Solution with Balsub using just the vectores ADD and REMOVE
	solutionFound = BalsubLast(targetSum);

	cEnd = wall_time();
	
	

	// Checking the solution
	for(i=1,aux=0; i<=n; i++)
		if (X[i]) aux += W[i];

	fprintf(stdout, "Solution: %llu=%llu for capacity %llu, n=%lu\n\n", (unsigned long long)solutionFound, (unsigned long long)aux, (unsigned long long)targetSum, (unsigned long)n);
	fprintf(stdout, "Ticks: \t\t%lf seg\n", (cEnd-cStart));
	
	// Alg, n, solution, capacity, seg
	fprintf(stderr, "BalsubLast,%llu,%llu,%llu,%lf,%u\n", (unsigned long long)n, (unsigned long long)solutionFound, (unsigned long long)targetSum, (cEnd-cStart), tries);

	// Free other memories
	free(&W[1]);
	free(&X[1]);

	exit(0);
}
