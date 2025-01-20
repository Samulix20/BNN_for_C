#ifndef BNN_LAYER_UTILS_H
#define BNN_LAYER_UTILS_H

#include "../activations.h"

// Flat index utils
inline size_t flat_idx_2d (size_t i, size_t j, size_t jlen) {
    return (i * jlen + j);
}

inline size_t flat_idx_3d (  
	size_t i, size_t j, size_t t, 
    size_t jlen, size_t tlen
) {
    return (i * jlen * tlen + flat_idx_2d(j, t, tlen));
}

inline size_t flat_idx_4d (
	size_t i, size_t j, size_t t, size_t k,
	size_t jlen, size_t tlen, size_t klen
) {
	return (i * jlen * tlen * klen + flat_idx_3d(j, t, k, tlen, klen));
}

#if PROFILING_MODE == 2
	uint64 num_bnn_mac = 0;
	uint64 num_bnn_add = 0;
#endif


__attribute__((always_inline))
inline Iop_t __fx_bnn_mac(Iop_t q_sigma, Iop_t q_mu, Iop_t q_x, Iop_t acc, Scale_t S) {
	
	const uint8 sgen = (31 - S) & 0b11111;
	const uint8 ilow = S & 0b111 ;
	const uint8 ihigh = (S >> 3) & 0b11;
	
	asm volatile(
		".insn r CUSTOM_0, 0, %[sgen], t6, x0, x0\n"
		".insn r4 CUSTOM_1, %[ilow], %[ihigh], t6, t6, %[si], %[mu]\n"
		".insn r4 CUSTOM_1, 0, 0, %[acc], t6, %[x], %[acc]\n"
		: [acc] "+r" (acc)
		: [si] "r" (q_sigma), [mu] "r" (q_mu), [x] "r" (q_x), [sgen] "i" (sgen), [ilow] "i" (ilow), [ihigh] "i" (ihigh)
		: "t6"
	);
	return acc;
}

__attribute__((always_inline))
inline Iop_t __fx_bnn_add(Iop_t q_sigma, Iop_t q_mu, Iop_t acc, Scale_t S) {
	
	const uint8 sgen = (31 - S) & 0b11111;
	const uint8 ilow = S & 0b111 ;
	const uint8 ihigh = (S >> 3) & 0b11;

	asm volatile(
		".insn r CUSTOM_0, 0, %[sgen], t6, x0, x0\n"
		".insn r4 CUSTOM_1, %[ilow], %[ihigh], t6, t6, %[si], %[mu]\n"
		"add %[acc], %[acc], t6"
		: [acc] "+r" (acc)
		: [si] "r" (q_sigma), [mu] "r" (q_mu), [sgen] "i" (sgen), [ilow] "i" (ilow), [ihigh] "i" (ihigh)
		: "t6"
	);
	return acc;
}


inline Iop_t bnn_mac(Iop_t q_sigma, Iop_t q_mu, Iop_t q_x, Iop_t acc, Scale_t S) {

	#if PROFILING_MODE == 2
		num_bnn_mac++;
	#endif

	#if BNN_INTERNAL_GEN == 2
		return __fx_bnn_mac(q_sigma, q_mu, q_x, acc, S);
	#else
		Iop_t w = get_weight(q_sigma, q_mu, S);
		acc += w * q_x;
		return acc;
	#endif
}


inline Iop_t bnn_add(Iop_t q_sigma, Iop_t q_mu, Iop_t acc, Scale_t S) {
	
	#if PROFILING_MODE == 2
		num_bnn_add++;
	#endif

	#if BNN_INTERNAL_GEN == 2
		return __fx_bnn_add(q_sigma, q_mu, acc, S);
	#else
		Iop_t q_bias = get_weight(q_sigma, q_mu, S);
		return acc + q_bias;
	#endif
}

#endif

