/*
GNU GENERAL PUBLIC LICENSE - Version 3

Copyright (c) 2018 Vitor Curtis <vitorcurtis@gmail.com>, Carlos Sanches <alonso@ita.br>
*/

#pragma once

#include "main.h"
#include "Alg3.c"

#ifndef ALG8_C
#define ALG8_C

void Alg8() {
	_N_TYPE i, add;
	_C_TYPE j;

	// Searching for optimal solution = remaining capacity
	if (flag) {
		if (s!=c) {												// Not exact
			j=s;												// Start from the last capacity found (s)
			s=s+Wmax; i=n;
			while (i>=1 && j+Wmax>c && j>=Wmin) {
				while (j+W[i]<c && i>=1) i--;					// Eliminating small items which don't catch up c

				for (;i>=1 && j+W[i]>=c && j>=Wmin; j--) {		// Searching j
					if (G[j]<i && j+W[i]<s) {					// Write condition and better solution
						add = i;
						s = j+W[i];
					}
				}
			}

			while (j<Wmin && i>=1 && W[i]<c) i--;				// Searching for solution W[i]>c
			if (j<Wmin && i>=1 && W[i]<s) {						// Better solution
				add = i;
				s = W[i];
			}

			X[add] = 1;
			Alg3( s-W[add] );
		} else {
			Alg3(c);
		}
		
		// Make the complement
		for(i=1; i<=n; ++i) X[i] = !X[i];
		s = sumW - s;

	// Searching for optimal solution = original capacity
	} else {
		Alg3(s);
	}
}

#endif // ALG8_C
