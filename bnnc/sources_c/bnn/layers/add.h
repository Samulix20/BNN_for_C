#include "utils.h"

inline void add_3D(
    Data_t* a, Data_t* b,
    size_t ilen, size_t jlen, size_t tlen,
    Data_t* output
) {
    size_t idx;

    for(size_t i = 0; i < ilen; i++) {
        for(size_t j = 0; j < jlen; j++) {
            for(size_t t = 0; t < tlen; t++) {
                idx = flat_idx_3d(i, j, t, jlen, tlen);
                Iop_t _a = (Iop_t) a[idx];
                Iop_t _b = (Iop_t) b[idx];
                output[idx] = (Data_t) (_a + _b);
            }
        }
    }
}

