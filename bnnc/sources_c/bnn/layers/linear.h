#include "utils.h"

inline void bnn_linear(
	Sigma_t* m_q_sigma,
	Mu_t* m_q_mu,
	Bias_t* mu_bias,
	Bias_Sigma_t* sigma_bias,
	Data_t* v_q_input,
	Data_t* v_q_ouput,
	Scale_t S,
	size_t ilen, 
	size_t jlen,
	enum Activation_Id f_act
) {
	Iop_t max = 0;
	Iop_t tmp_o[ilen];

	for(size_t i = 0; i < ilen; i++) {

		Iop_t acc = 0;

		for(size_t j = 0; j < jlen; j++) {
			size_t m_idx = flat_idx_2d(i, j, jlen);
			Iop_t q_sigma = (Iop_t) m_q_sigma[m_idx];
			Iop_t q_mu = (Iop_t) m_q_mu[m_idx]; 
			Iop_t q_x = (Iop_t) v_q_input[j];

			Iop_t w = get_weight(q_sigma, q_mu, S);

			acc += w * q_x; 
		}

		Iop_t q_mu_bias = (Iop_t) mu_bias[i];
		Iop_t q_sigma_bias = (Iop_t) sigma_bias[i];
		Iop_t q_bias = get_weight(q_sigma_bias, q_mu_bias, S);
		
		Iop_t q_acc = acc >> S;
		Iop_t q_o = (Data_t) (q_acc + q_bias);
		
		if (f_act == ReLU_ID) {
			v_q_ouput[i] = ReLU(q_o);
		} 
		else if (f_act == Softmax_ID) {
			// Find max
			if(q_o > max || i == 0) max = q_o;
			tmp_o[i] = q_o;
		}
	}

	if (f_act == Softmax_ID) {
		softmax(tmp_o, v_q_ouput, max, S, ilen);
	}
}
