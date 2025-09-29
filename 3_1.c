#include <stdio.h>
#include <mpi.h>
// Función que encuentra el bin adecuado para un valor dadoo
int find_bin(float value, float bin_maxes[], int bin_count, float min_meas) {
    for (int b = 0; b < bin_count; b++) {
        if (b == 0) {
            if (value >= min_meas && value < bin_maxes[b]) return b;
        } else {
            if (value >= bin_maxes[b - 1] && value < bin_maxes[b]) return b;
        }
    }
    return bin_count - 1;  // En caso de ser el valor máximo
}

// Función principal que crea el histograma
void create_histogram(float data[], int data_count, float min_meas, float max_meas, int bin_count) {
    float bin_width = (max_meas - min_meas) / bin_count;
    float bin_maxes[bin_count];
    int bin_counts[bin_count];

    // Inicializar los límites de los bins y los contadores en cero
    for (int b = 0; b < bin_count; b++) {
        bin_maxes[b] = min_meas + bin_width * (b + 1);
        bin_counts[b] = 0;
    }

    // Clasificar cada valor en el bin adecuado
    for (int i = 0; i < data_count; i++) {
        int bin = find_bin(data[i], bin_maxes, bin_count, min_meas);
        bin_counts[bin]++;
    }

    // Imprimir el histograma
    printf("Histograma:\n");
    for (int b = 0; b < bin_count; b++) {
        printf("Bin %d (%.1f - %.1f): %d\n", b, 
               b == 0 ? min_meas : bin_maxes[b - 1], 
               bin_maxes[b], bin_counts[b]);
    }
}

int main() {
    // Datos de ejemplo
    float data[] = {1.3, 2.9, 0.4, 0.3, 1.3, 4.4, 1.7, 0.4, 3.2, 0.3,
                    4.9, 2.4, 3.1, 4.4, 3.9, 0.4, 4.2, 4.5, 4.9, 0.9};
    int data_count = sizeof(data) / sizeof(data[0]);
    float min_meas = 0.0;
    float max_meas = 5.0;
    int bin_count = 5;

    // Crear el histograma
    create_histogram(data, data_count, min_meas, max_meas, bin_count);
    
    return 0;
}