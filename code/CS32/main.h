/*
GNU GENERAL PUBLIC LICENSE - Version 3

Copyright (c) 2018 Vitor Curtis <vitorcurtis@gmail.com>, Carlos Sanches <alonso@ita.br>
*/

#pragma once

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#ifndef MAIN_H
#define MAIN_H

// Machine architecture
#define ARCH				32				// (in bits)

// Max value of N (lower better)
#define N_MAX				16				// (in bits)
// Max value of each Wi
#define WI_MAX				32				// (in bits)
// Max value of SUM Wi and C
#define SUM_WI_MAX			64				// (in bits)



#if (UCHAR_MAX == 0xff)
#define U8BITS unsigned char
#else
#error Define correct type of 8 bits
#endif

#if (USHRT_MAX == 0xffff)
#define U16BITS unsigned short
#else
#error Define correct type of 16 bits
#endif

#if (UINT_MAX == 0xffffffff)
#define U32BITS unsigned int
#else
#error Define correct type of 32 bits
#endif

#if (ULLONG_MAX == 0xffffffffffffffffULL)
#define U64BITS unsigned long long
#else
#error Define correct type of 64 bits
#endif

// Architecture of the machine
#if ARCH==16
	#define _ARCH_TYPE U16BITS
#elif ARCH==32
	#define _ARCH_TYPE U32BITS
#elif ARCH==64
	#define _ARCH_TYPE U64BITS
#else
#error Wrong value of ARCH
#endif

// Type of vetor G elements
#if N_MAX==8
	#define _N_TYPE U8BITS
#elif N_MAX==16
	#define _N_TYPE U16BITS
#elif N_MAX==32
	#define _N_TYPE U32BITS
#elif N_MAX==64
	#define _N_TYPE U64BITS
#else
#error Wrong value of N_MAX
#endif

// Type of each Wi (Vector W)
#if WI_MAX==16
	#define _WI_TYPE U16BITS
#elif WI_MAX==32
	#define _WI_TYPE U32BITS
#elif WI_MAX==64
	#define _WI_TYPE U64BITS
#else
#error Wrong value of WI_MAX
#endif

// Type of SUM Wi and C
#if SUM_WI_MAX==16
	#define _C_TYPE U16BITS
#elif SUM_WI_MAX==32
	#define _C_TYPE U32BITS
#elif SUM_WI_MAX==64
	#define _C_TYPE U64BITS
#else
#error Wrong value of SUM_WI_MAX
#endif



#ifdef _MSC_VER
#define STIN static __inline
#else
#define STIN static inline
#endif

#define mod_q(a) ((a)&(ARCH-1))
#define max(a,b) ((b)>(a)?(b):(a))
#define min(a,b) ((b)<(a)?(b):(a))



unsigned char* X;						// Solution vector
_N_TYPE* G=NULL;						// G vector for CPU
_WI_TYPE* W=NULL;						// List of integers

_C_TYPE targetSum;						// Original sum to search
_C_TYPE c;								// Sum to search
_C_TYPE s;								// Best solution known
_N_TYPE n;								// Amount of integers in the List

_C_TYPE sumW;							// Sum of all integers
_WI_TYPE Wmin, Wmax;					// lightest and heaviest integers in W
_C_TYPE t;								// Used to calculate Toth
_N_TYPE Wmin_i, Wmax_i;					// lightest and heaviest indexes
_C_TYPE sum1_i, sumi1_n;				// Complementary sums
char flag;

#define printError(funcName) fprintf(stderr, "Error: %s at %s, line %d!\n", funcName, __FILE__, __LINE__)

STIN void calc_SumW_Wmin_Wmax() {
	_N_TYPE i;

	// Calculating sum of Wi, lightest and heaviest items
	sumW=0; Wmin_i=1; Wmax_i=1;

	for(i=1; i<=n; i++) {
		if (W[i]<W[Wmin_i]) Wmin_i=i;		// Finding the lightest integer
		if (W[i]>W[Wmax_i]) Wmax_i=i;		// Finding the heaviest integer

		sumW += W[i];						// Sum of all integers
	}

	Wmin = W[Wmin_i];
	Wmax = W[Wmax_i];
}

STIN void readFile(FILE *file) {
	_N_TYPE i;
	unsigned long long data;

	// Reading amount of integers (n) and targetSum (c)
	fscanf(file, "%llu\n", &data); n = (_N_TYPE)data;
	fscanf(file, "%llu\n", &data); targetSum = (_C_TYPE)data;
	c = targetSum;

	// Allocating vector space
	W = (_WI_TYPE*) malloc(n*sizeof(_WI_TYPE));
	if (W==NULL) {
		fprintf(stderr, "Error: malloc(%llu) at %s, line %d!\n", (unsigned long long)n*sizeof(_WI_TYPE), __FILE__, __LINE__);
		exit(-1);
	}
	W = &W[-1];			// Fixing indexes

	// Allocating solution space
	X = (unsigned char*) calloc(n, sizeof(unsigned char));
	if (X==NULL) {
		fprintf(stderr, "Error: malloc(%llu) at %s, line %d!\n", (unsigned long long)n*sizeof(unsigned char), __FILE__, __LINE__);
		exit(-1);
	}
	X = &X[-1];			// Fixing indexes

	// Filling the vector
	for(i=1; i<=n; i++) {
		fscanf(file, "%llu\n,", &data);
		W[i] = (_WI_TYPE)data;
	}
	fclose(file);
}

STIN void init(int argc, char *argv[]) {
	FILE *fsss;

	/*
	* Command Syntax
	*/
	if (argc<2) {
		fprintf (stdout, "Inform the file that has the integers and target sum in SS format.\n");
		fprintf (stdout, "\nUse: %s file.ss\n", argv[0]);
		fprintf (stdout, "\nParameters:\n");
		fprintf (stdout, "file - path of the SS file\n");
		exit(-1);
	}

	fsss = fopen(argv[1], "r");

	if(fsss == NULL) {
		fprintf(stderr, "Error: File not found %s!\n", argv[1]);
		exit(-1);
	}

	readFile(fsss);
}

/*
 * Sorting comparisons
 */
STIN int compareAscending(const void * a, const void * b) {
    return (*(_WI_TYPE*)a < *(_WI_TYPE*)b)?-1:(*(_WI_TYPE*)a == *(_WI_TYPE*)b)?0:1;
}

STIN int compareDescending(const void * a, const void * b) {
    return (*(_WI_TYPE*)a > *(_WI_TYPE*)b)?-1:(*(_WI_TYPE*)a == *(_WI_TYPE*)b)?0:1;
}

/*
 * Time related functions
 */
#ifdef _WIN32
	#include <windows.h>
	#if defined(_MSC_VER) || defined(_MSC_EXTENSIONS)
		#define DELTA_EPOCH_IN_MICROSECS  11644473600000000Ui64
	#else
		#define DELTA_EPOCH_IN_MICROSECS  11644473600000000ULL
	#endif
	
struct timezone {
	int  tz_minuteswest; /* minutes W of Greenwich */
	int  tz_dsttime;     /* type of dst correction */
};

int gettimeofday(struct timeval *tv, struct timezone *tz) {
	FILETIME ft;
	unsigned __int64 tmpres = 0;
	static int tzflag;

	if (NULL != tv) {
		GetSystemTimeAsFileTime(&ft);

		tmpres |= ft.dwHighDateTime;
		tmpres <<= 32;
		tmpres |= ft.dwLowDateTime;

		/*converting file time to unix epoch*/
		tmpres -= DELTA_EPOCH_IN_MICROSECS; 
		tmpres /= 10;  /*convert into microseconds*/
		tv->tv_sec = (long)(tmpres / 1000000UL);
		tv->tv_usec = (long)(tmpres % 1000000UL);
	}

	if (NULL != tz) {
		if (!tzflag) {
			_tzset();
			tzflag++;
		}
		tz->tz_minuteswest = _timezone / 60;
		tz->tz_dsttime = _daylight;
	}

	return 0;
}

#elif __unix__
	#include <sys/time.h>
#endif

double wall_time(void) {
  struct timeval tv;
  struct timezone tz;

  gettimeofday(&tv, &tz);
  return(tv.tv_sec + tv.tv_usec/1000000.0);
}

#endif // MAIN_H
