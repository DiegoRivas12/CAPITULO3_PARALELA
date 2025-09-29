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

// Función para crear el histograma de datos locales
void create_local_histogram(float data[], int data_count, float min_meas, float max_meas, 
                            int bin_count, int bin_counts[]) {
    float bin_width = (max_meas - min_meas) / bin_count;
    float bin_maxes[bin_count];

    // Inicializar los límites de los bins
    for (int b = 0; b < bin_count; b++) {
        bin_maxes[b] = min_meas + bin_width * (b + 1);
        bin_counts[b] = 0;  // Inicializar los contadores en cero
    }

    // Clasificar cada valor en el bin adecuado
    for (int i = 0; i < data_count; i++) {
        int bin = find_bin(data[i], bin_maxes, bin_count, min_meas);
        bin_counts[bin]++;
    }
}

int main(int argc, char* argv[]) {
    int my_rank, comm_sz;
    int data_count = 20;
    float min_meas = 0.0;
    float max_meas = 5.0;
    int bin_count = 5;
    float data[] = {1.3, 2.9, 0.4, 0.3, 1.3, 4.4, 1.7, 0.4, 3.2, 0.3,
                    4.9, 2.4, 3.1, 4.4, 3.9, 0.4, 4.2, 4.5, 4.9, 0.9};//20 datos
    //float data[] = {1.3, 2.9, 0.4, 0.3, 1.3, 4.4, 1.7, 0.4, 3.2, 0.3,4.9, 2.4, 3.1, 4.4, 3.9, 0.4, 4.2, 4.5, 4.9, 0.9};

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    // Dividir los datos entre los procesos
    int local_data_count = data_count / comm_sz;
    //printf("local data %d",local_data_count);
    float local_data[local_data_count];//A cada  proceso se le daraun subarreglo del mismo tamano que4 conforman el arreglo original
    /*MPI_Scatter(
    send_data,      // Dirección del arreglo a enviar (solo válido en el proceso raíz)
    send_count,     // Número de elementos a enviar a cada proceso
    send_datatype,  // Tipo de dato de cada elemento a enviar
    recv_data,      // Dirección del buffer donde se almacenarán los datos recibidos
    recv_count,     // Número de elementos que cada proceso recibirá
    recv_datatype,  // Tipo de dato de cada elemento recibido
    root,           // ID del proceso raíz (típicamente 0)
    communicator    // Comunicador, generalmente MPI_COMM_WORLD
);*/
    MPI_Scatter(data, local_data_count, MPI_FLOAT, local_data, local_data_count, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Cada proceso calcula su histograma local
    int local_bin_counts[bin_count];
    create_local_histogram(local_data, local_data_count, min_meas, max_meas, bin_count, local_bin_counts);

    // Recolectar y sumar los resultados en el proceso 0
    int global_bin_counts[bin_count];
    /*
    MPI_Reduce(
    send_data,      // Dirección de los datos locales a enviar
    recv_data,      // Dirección donde se almacenará el resultado combinado (solo en el proceso raíz)
    count,          // Número de elementos en el buffer `send_data`
    datatype,       // Tipo de dato de los elementos
    operation,      // Operación de reducción (por ejemplo, MPI_SUM para suma)
    root,           // ID del proceso raíz donde se almacenará el resultado
    communicator    // Comunicador, generalmente MPI_COMM_WORLD
);
    */
    MPI_Reduce(local_bin_counts, global_bin_counts, bin_count, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // El proceso 0 imprime el histograma completo
    if (my_rank == 0) {
        float bin_width = (max_meas - min_meas) / bin_count;
        printf("Histograma:\n");
        for (int b = 0; b < bin_count; b++) {
            printf("Bin %d (%.1f - %.1f): %d\n", b, 
                   b == 0 ? min_meas : min_meas + bin_width * b, 
                   min_meas + bin_width * (b + 1), global_bin_counts[b]);
        }
    }

    MPI_Finalize();
    return 0;
}
