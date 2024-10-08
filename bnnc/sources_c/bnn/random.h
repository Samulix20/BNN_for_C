#ifndef C_BNN_RAND_H
#define C_BNN_RAND_H

#include "types.h"

#define DEFAULT_SEED 0xDEADBEEF

// https://en.wikipedia.org/wiki/Xorshift
inline uint32 xorshift32(uint32 seed) {
    uint32 x = seed;
	x ^= x << 13;
	x ^= x >> 17;
	x ^= x << 5;
	return x;
}

// Deprecated
// Bernoulli sample scaled at S
inline Iop_t bernoulli_sample(Scale_t S, Sigma_t q, Mu_t p) {
    extern uint32 seed;
    // seed S = 32
    seed = xorshift32(seed);
    Mu_t scaled_sample =  (Mu_t) (seed >> (32 - S));
    if (p > scaled_sample) return (Iop_t) q;
    else return 0;
}

// Uniform sample scaled at S
inline Iop_t uniform_sample(Scale_t S) {
    extern uint32 seed;
    // Seed can be iterpreted as scaled at S = 32
    seed = xorshift32(seed);
    return (Iop_t) (seed >> (32 - S));
}

// Normal sample scaled at S using clt approximation
inline Iop_t clt_normal_sample(Scale_t S) {
    // Scaled at S
    Iop_t acc = 0;
    for(size_t i = 0; i < 12; i++) {
        acc += uniform_sample(S);
    }
    // Center value
    acc -= (6 << S); // Scale 6 to S
    return acc;
}

/*
	#######################
	WEIGHT SAMPLING WRAPPER
	#######################
*/ 

inline Iop_t generator(Scale_t S) {
    Iop_t g;

    #if BNN_INTERNAL_GEN == 0
		g = clt_normal_sample(S);
	#elif BNN_INTERNAL_GEN == 1
        g = uniform_sample(S);
    #elif BNN_INTERNAL_GEN == 2
        #include <riscv/custom.h>
        #if 12 > BNN_SCALE_FACTOR
            g = gen_num() >> (12 - S);
        #else
            g = gen_num() << (S - 12);
        #endif 
    #endif

    return g;
}

inline Iop_t get_weight(Iop_t q_sigma, Iop_t q_mu, Scale_t S) {
    Iop_t w;

    #if BNN_INTERNAL_GEN == 0 || BNN_INTERNAL_GEN == 1 || BNN_INTERNAL_GEN == 2
        Iop_t u = generator(S); 
        w = ((q_sigma * u) >> S) + q_mu;
    #else
        w = bernoulli_sample(S, q_sigma, q_mu);
    #endif

    return w;
}

#endif
