#ifndef BNN_TYPES_H
#define BNN_TYPES_H

#include "../bnn_config.h"

// Default config if not defined
/*
#ifndef BNN_CONFIG_H
#define BNN_CONFIG_H

#define BNN_SIGMA_DT        int32
#define BNN_MU_DT           int32
#define BNN_BIAS_DT         int32
#define BNN_BIAS_SIGMA_DT   int32

#define BNN_DATA_DT         int32

#define BNN_SCALE_FACTOR 12
#define BNN_INTERNAL_GEN 0

#define BNN_MC_PASSES 100
#endif
*/

#include <stddef.h>
#include <stdint.h>

// Standard type sizes
typedef int8_t int8;
typedef uint8_t uint8;

typedef int16_t int16;
typedef uint16_t uint16;

typedef int32_t int32;
typedef uint32_t uint32;

typedef int64_t int64;
typedef uint64_t uint64;

// Fixed point operations
#define ITOFIX(i, n, dt)			(((dt) (i)) << (n))
#define FTOFIX(f, n, dt)			((dt) ((f) * (1 << (n))))
#define FIXTOF(f, n)				(((float) (f)) / (1 << (n)))

#define MULFIX(a, b, n)				(((a) * (b)) >> (n))
#define _MULFIX(a, b, n, dt, dt2)	((dt) (((dt2) (a) * (dt2) (b)) >> (n)))

#define DIVFIX(a, b, n)				(((a) << (n)) / (b))
#define _DIVFIX(a, b, n, dt, dt2)	((dt) (((dt2) (a) << (n)) / (dt2) (b)))

typedef BNN_SIGMA_DT Sigma_t;
typedef BNN_MU_DT Mu_t;
typedef BNN_BIAS_DT Bias_t;
typedef BNN_BIAS_SIGMA_DT Bias_Sigma_t;

typedef BNN_DATA_DT Data_t;

// Data type for the internal operations of the library
typedef int32 Iop_t;

// Represents negative power of 2 exponents
typedef int8 Scale_t;

// High precission fixed point data type for softmax calculations
typedef int32 Softmax_t;
typedef uint32 u_Softmax_t;
// Fixed bits for softmax data type
#define S_Softmax 28

#endif
