#ifndef C_BNN_RAND_H
#define C_BNN_RAND_H

#include "types.h"

#define DEFAULT_SEED 0xDEADBEEF
uint32 bnn_random_seed = DEFAULT_SEED;

// https://en.wikipedia.org/wiki/Xorshift
inline uint32 xorshift32(uint32 seed) {
    uint32 x = seed;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return x;
}

#ifdef FLOATING_TYPES

    __attribute__((always_inline))
    inline Iop_t uniform_sample() {
        bnn_random_seed = xorshift32(bnn_random_seed);
        return (Iop_t) (bnn_random_seed >> 1) / (1 << 31);
    }

#else

    // Fixed point samplers and generators
    // Uniform sample scaled at S
    __attribute__((always_inline))
    inline Iop_t uniform_sample(Scale_t S) {
        // Seed can be iterpreted as scaled at S = 32
        bnn_random_seed = xorshift32(bnn_random_seed);
        return (Iop_t) (bnn_random_seed >> (32 - S));
    }

#endif

// Bernoulli sample scaled at S
__attribute__((always_inline))
inline Iop_t bernoulli_sample(Scale_t S, Iop_t p) {
    Iop_t scaled_sample = uniform_sample(S);
    if (p > scaled_sample) return 1;
    else return 0;
}

// Normal sample scaled at S using clt approximation
__attribute__((always_inline))
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

#endif
