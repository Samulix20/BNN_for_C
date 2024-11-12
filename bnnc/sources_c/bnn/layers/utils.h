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

Iop_t bnn_mac(Iop_t q_sigma, Iop_t q_mu, Iop_t q_x, Iop_t acc, Scale_t S) {
	Iop_t w = get_weight(q_sigma, q_mu, S);
	acc += w * q_x;
	return acc;
}


Iop_t bnn_add(Iop_t q_sigma, Iop_t q_mu, Iop_t acc, Scale_t S) {
	Iop_t q_bias = get_weight(q_sigma, q_mu, S);
	return acc + q_bias;
}

#endif

