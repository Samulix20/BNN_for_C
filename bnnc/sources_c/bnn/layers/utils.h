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

#endif

