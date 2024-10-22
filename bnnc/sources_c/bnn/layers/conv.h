#include "utils.h"

inline void bnn_conv2D (
	Data_t* t_q_input, 
    size_t ilen, size_t jlen, size_t tlen,
	size_t num_filters, size_t kernel_size,
	size_t pad_i, size_t pad_j,
	size_t stride_i, size_t stride_j,
	Mu_t* mu_kernels, 
    Sigma_t* sigma_kernels, 
    Bias_t* mu_bias,
	Bias_Sigma_t* sigma_bias, 
	Data_t* t_q_output,
    enum Activation_Id f_act,
	Scale_t S
) {
	// Padding same, same input and output size
	const size_t out_ilen = ((ilen + 2 * pad_i - 1 * (kernel_size - 1) - 1) / stride_i) + 1;
	const size_t out_jlen = ((jlen + 2 * pad_j - 1 * (kernel_size - 1) - 1) / stride_j) + 1;
	const size_t out_tlen = num_filters;

	size_t idx;

	// for each filter == for each output channel
	for(size_t t = 0; t < out_tlen; t++) {

		idx = flat_idx_4d(t,0,0,0, kernel_size, kernel_size, tlen);

		Mu_t* t_q_mu = mu_kernels + idx;
		Sigma_t* t_q_sigma = sigma_kernels + idx;
		
		Iop_t q_mu_bias = (Iop_t) mu_bias[t]; 
		Iop_t q_sigma_bias = (Iop_t) sigma_bias[t]; 
		Iop_t q_bias = get_weight(q_sigma_bias, q_mu_bias, S);

		// for each submatrix
		size_t i, j;
		for(i = 0; i < out_ilen; i++) {
			for(j = 0; j < out_jlen; j++) {
				
				Iop_t acc = 0;
				
				// for each element of 2D kernel and channels
				size_t ki, kj, tt;
				for(ki = 0; ki < kernel_size; ki++) {
					for(kj = 0; kj < kernel_size; kj++) {
						for(tt = 0; tt < tlen; tt++) {

							// Input index strides i,j
							size_t ii = (i * stride_i) + ki;
							size_t jj = (j * stride_j) + kj;

							// Check left pad element
							if ((ii < pad_i) || (jj < pad_j)) {
								// pad 0 then v = 0
								// If v == 0 then acc += 0, can ignore
								continue;
							}

							// Fix left pad offset
							ii -= pad_i;
							jj -= pad_j;

							// If out of original matrix is right padding
							if ((ii >= ilen) || (jj >= jlen)) {
								// Ignore
								continue;
							}

							// Sample weight
							idx = flat_idx_3d(ki, kj, tt, kernel_size, tlen);
							
							Iop_t q_sigma = (Iop_t) t_q_sigma[idx];
							Iop_t q_mu = (Iop_t) t_q_mu[idx];

							Iop_t w = get_weight(q_sigma, q_mu, S);

							idx = flat_idx_3d(ii, jj, tt, jlen, tlen);
							Iop_t q_x = (Iop_t) t_q_input[idx];
							acc += w * q_x;
						}
					}
				}

				Iop_t q_acc = acc >> S;
				Data_t q_o = (Data_t) (q_acc + q_bias);

				idx = flat_idx_3d(i, j, t, out_jlen, out_tlen);
				
                // Apply f_act
				if (f_act == ReLU_ID) {
					q_o = ReLU(q_o);
				}

				t_q_output[idx] = q_o;
			}
		}
	}
}
