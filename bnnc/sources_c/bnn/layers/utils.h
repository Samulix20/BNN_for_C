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

// Internal kernels

__attribute__((always_inline))
inline Iop_t sample_weight_clt(Iop_t q_sigma, Iop_t q_mu, Scale_t S) {
	Iop_t sample = clt_normal_sample(S);
	Iop_t aux = q_sigma * sample;
	#ifndef FLOATING_TYPES
		aux >>= S;
	#endif
	return aux + q_mu;
}

__attribute__((always_inline))
inline Iop_t sample_weight_uniform(Iop_t q_sigma, Iop_t q_mu, Scale_t S) {
	Iop_t sample = uniform_sample(S);
	Iop_t aux = q_sigma * sample;
	#ifndef FLOATING_TYPES
		aux >>= S;
	#endif
	return aux + q_mu;
}

__attribute__((always_inline))
inline Iop_t sample_weight_bernoulli(Iop_t q_sigma, Iop_t q_mu, Scale_t S) {
	Iop_t sample = bernoulli_sample(S, q_mu);
	Iop_t aux = q_sigma * sample;
	#ifndef FLOATING_TYPES
		aux >>= S;
	#endif
	return aux;
}

__attribute__((always_inline))
inline Iop_t sample_weight(Iop_t q_sigma, Iop_t q_mu, Scale_t S) {
	#if BNN_INTERNAL_GEN == 0
		return sample_weight_clt(q_sigma, q_mu, S);
	#elif BNN_INTERNAL_GEN == 1
		return sample_weight_uniform(q_sigma, q_mu, S);
	#elif BNN_INTERNAL_GEN == 3
		return sample_weight_bernoulli(q_sigma, q_mu, S);
	#endif
}

__attribute__((always_inline))
inline Iop_t bnn_mac(Iop_t q_sigma, Iop_t q_mu, Iop_t q_x, Iop_t acc, Scale_t S) {
	
	#if BNN_INTERNAL_GEN == 0 || BNN_INTERNAL_GEN == 1 || BNN_INTERNAL_GEN == 3
		return acc + sample_weight(q_sigma, q_mu, S) * q_x;
	#elif BNN_INTERNAL_GEN == 2
		return __fx_bnn_mac(q_sigma, q_mu, q_x, acc, S);
	#endif
}

__attribute__((always_inline))
inline Iop_t bnn_add(Iop_t q_sigma, Iop_t q_mu, Iop_t acc, Scale_t S) {
	
	#if BNN_INTERNAL_GEN == 0 || BNN_INTERNAL_GEN == 1 || BNN_INTERNAL_GEN == 3
		return acc + sample_weight(q_sigma, q_mu, S);
	#elif BNN_INTERNAL_GEN == 2
		return __fx_bnn_add(q_sigma, q_mu, acc, S);
	#endif
}

#endif

