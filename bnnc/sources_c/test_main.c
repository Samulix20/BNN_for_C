// Model headers and config include

#ifndef PROFILING_MODE
    #define PROFILING_MODE 0
#endif

#include "bnn_model_weights.h"
#include "bnn_model.h"
#include "test_data.h"

#include <stdio.h>

int main() {
    size_t output_size = bnn_model_num_classes;

    // Print CSV header for output
    // input, mcpass, class0, class1, ...
    printf("input,mcpass,");
    for(size_t i = 0; i < output_size; i++) {
        printf("class%i", i);
        if (i != output_size - 1) printf(",");
    }
    printf("\n");

    for(size_t i = 0; i < NUM_DATA; i++) {
        for(size_t j = 0; j < BNN_MC_PASSES; j++) {

            #if PROFILING_MODE == 1

            // Profiler version unsing riscv counters
            #include <riscv/profiler/external.h>
            start_external_counter(0);
            Data_t* output_p = bnn_model_inference(data_matrix + FEATURES_PER_DATA*i);
            stop_external_counter(0);

            #else

            Data_t* output_p = bnn_model_inference(data_matrix + FEATURES_PER_DATA*i);

            #endif

            // Print prediction
            printf("%i, %i, ", i, j);
            for(size_t k = 0; k < output_size; k++) {
                
                #if PROFILING_MODE == 0
                    #ifdef FLOATING_TYPES
                        printf("%f", output_p[k]);
                    #else
                        printf("%f", FIXTOF(output_p[k], BNN_SCALE_FACTOR));
                    #endif
                #else
                    printf("%i", output_p[k]);
                #endif
                
                if (k != output_size - 1) printf(", "); 
            }
            printf("\n");
        }
    }

    #if PROFILING_MODE == 2
        extern uint64 num_bnn_mac;
        extern uint64 num_bnn_add;

        printf("BNN MAC %llu\n", num_bnn_mac);
        printf("BNN ADD %llu\n", num_bnn_add);
    #endif
}

