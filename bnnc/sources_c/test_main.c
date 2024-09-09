// Model headers and config include

#include "bnn_model_weights.h"
#include "bnn_model.h"

#include <stdio.h>

// TODO REMOVE ONLY FOR TESTING
#define FEATURES_PER_DATA 1
#define NUM_DATA 1
Data_t data_matrix[500000];


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
            
            Data_t* output_p = bnn_model_inference(data_matrix + FEATURES_PER_DATA*i);
            
            // Print prediction
            printf("%i, %i, ", i, j);
            for(size_t k = 0; k < output_size; k++) {
                printf("%f", FIXTOF(output_p[k], S_Softmax));
                if (k != output_size - 1) printf(", "); 
            }
            printf("\n");
        }
    }
}

