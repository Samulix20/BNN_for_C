#ifndef BNN_LAYERS_H
#define BNN_LAYERS_H

#include "activations.h"

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

void bnn_conv2D_valid_ReLU (
	Data_t* t_q_input, 
    size_t ilen, size_t jlen, size_t tlen,
	size_t num_filters, size_t kernel_size,
	Mu_t* mu_kernels, 
    Sigma_t* sigma_kernels, 
    Bias_t* v_q_bias,
	Data_t* t_q_output,
    Scale_t S
);

void bnn_conv2D_same_ReLU (
    Data_t* t_q_input, 
    size_t ilen, size_t jlen, size_t tlen,
	size_t num_filters, size_t kernel_size,
	Mu_t* mu_kernels, 
    Sigma_t* sigma_kernels, 
    Bias_t* v_q_bias,
	Data_t* t_q_output,
    Scale_t S
);

void bnn_linear_ReLU(
	Sigma_t* m_q_sigma,
	Mu_t* m_q_mu,
	Bias_t* v_q_bias,
	Data_t* v_q_input,
	Data_t* v_q_ouput,
	Scale_t S,
	size_t ilen, 
	size_t jlen
);

void bnn_linear_Softmax(
	Sigma_t* m_q_sigma,
	Mu_t* m_q_mu,
	Bias_t* v_q_bias,
	Data_t* v_q_input,
	Softmax_t* v_q_ouput,
	Scale_t S,
	size_t ilen, 
	size_t jlen
);

void layer_max_pooling2D(
	Data_t* input,
	size_t ilen, size_t jlen, size_t tlen,
	size_t stride_i, size_t stride_j,
	Data_t* output
);


#endif
