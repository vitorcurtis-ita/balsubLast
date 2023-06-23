/*
GNU GENERAL PUBLIC LICENSE - Version 3

Copyright (c) 2018 Vitor Curtis <vitorcurtis@gmail.com>, Carlos Sanches <alonso@ita.br>
*/

#pragma once

#include "main.h"

#ifndef ALG3_C
#define ALG3_C

void Alg3(_C_TYPE s) {
	_C_TYPE j = s;

	while(j!=0) {
		X[G[j]] = 1;
		j = j - W[G[j]];
	}
}

#endif // ALG3_C
