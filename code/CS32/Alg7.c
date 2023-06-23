/*
GNU GENERAL PUBLIC LICENSE - Version 3

Copyright (c) 2018 Vitor Curtis <vitorcurtis@gmail.com>, Carlos Sanches <alonso@ita.br>
*/

#pragma once

#include "main.h"

#ifndef ALG7_C
#define ALG7_C

STIN void makeTrivialSolutionStart(_C_TYPE k) {
	_N_TYPE i=1;

	while (k != 0) {
		if (k >= W[i]) {
			k -= W[i];
			X[i] = 1;
		}
		i++;
	}
}

STIN void makeTrivialSolutionEnd(_C_TYPE k) {
	_N_TYPE i=n;

	while (k != 0) {
		if (k >= W[i]) {
			k -= W[i];
			X[i] = 1;
		}
		i--;
	}
}

// Return true if a solution was found
STIN unsigned int verifyTrivialSolutions() {
	_N_TYPE i;
	_C_TYPE fromBegin, fromEnd;		// Test for trivial solutions



	// If Wmin is higher than c (it only occours in remaining capacity)
	if (Wmin >= sumW - c) {
		fprintf(stdout, "Trivial remaining solution found!\n");
		s = Wmin;								// Solution
		c = sumW - c;							// Change to complement
		X[Wmin_i]=1;							// Vector solution
		goto Complement;
	}





	// Seach for a trivial solution = c
	fromBegin = fromEnd = c;

	for(i=1; i<=n && fromBegin && fromEnd; i++) {
		if (fromBegin >= W[i])
			fromBegin -= W[i];

		if (fromEnd >= W[n-i+1])
			fromEnd -= W[n-i+1];
	}

	if (!fromBegin || !fromEnd) {
		s = c;									// Solution
		fprintf(stdout, "\nTrivial solution found!\n");

		if (!fromBegin) {
			makeTrivialSolutionStart(s);
		} else if (!fromEnd) {
			makeTrivialSolutionEnd(s);
		}
		goto Complement;
	}

	// Set the best solution found
	t = c - min(fromBegin, fromEnd);



	// Seach for a trivial solution with complement
	fromBegin = fromEnd = sumW - c;

	for(i=1; i<=n && fromBegin && fromEnd; i++) {
		if (fromBegin >= W[i])
			fromBegin -= W[i];

		if (fromEnd >= W[n-i+1])
			fromEnd -= W[n-i+1];
	}

	if (!fromBegin || !fromEnd) {
		s = sumW - c;							// Solution
		c = sumW - c;							// Change to complement
		fprintf(stdout, "\nTrivial solution found (2)!\n");

		if (!fromBegin) {
			makeTrivialSolutionStart(s);
		} else if (!fromEnd) {
			makeTrivialSolutionEnd(s);
		}
		goto Complement;
	}

	// No exact solution found
	return 0;

Complement:

	// If computing remainder, make the complement
	if (targetSum!=c) {
		for(i=1; i<=n; ++i)
			X[i] = !X[i];
		s = sumW - s;
	}

	return 1;
}

// Return true if a solution was found
int Alg7() {
	_C_TYPE j;



	// Sorting items
	qsort(&W[1], n, sizeof(_WI_TYPE), compareDescending);

	calc_SumW_Wmin_Wmax();

	// Searching trivial solutions
	if (verifyTrivialSolutions()) return 1;

	// Deciding which sum to search
	if (c > sumW/2) {
		flag = 1;
		c = sumW - c;
		t = c;
	} else {
		flag = 0;
	}



	// Allocating the G vector for CPU
	if (Wmin > c) {
		printError("malloc"); exit(-1);
	}
	G = (_N_TYPE*) malloc((c-Wmin+1)*sizeof(_N_TYPE));
	if (G == NULL) { printError("malloc"); exit(-1); }
	G = &G[-((int)Wmin)];		// Correction

	// Filling elementary combinations
	for(j=Wmin; j<=c; j++)
		G[j] = n+1;

	return 0;
}

#endif // ALG7_C
