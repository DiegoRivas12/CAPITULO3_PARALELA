#include <stdio.h>
#include <mpi.h>

int main(int argc, char* argv[]) {
    int my_rank, comm_sz;
    int local_sum = 1; // Cada proceso comienza con un valor local de 1a
    int global_sum = 0;

    // Inicializar MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    // Realizar reducción estructurada en árbol
    int step;
    for (step = 1; step < comm_sz; step *= 2) {
        if (my_rank % (2 * step) == 0) { // Si my_rank es par
            int partner = my_rank + step; // El compañero es el siguiente rango en el árbol
            if (partner < comm_sz) { // Verificar si el compañero está dentro del rango
                MPI_Recv(&local_sum, 1, MPI_INT, partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                local_sum += local_sum; // Sumar el valor recibido
            }
        } else {
            int partner = my_rank - step; // El compañero es el rango anterior en el árbol
            MPI_Send(&local_sum, 1, MPI_INT, partner, 0, MPI_COMM_WORLD);
            break; // Salir una vez que se complete la operación de envío
        }
    }

    // En la última ronda, necesitamos reunir los valores restantes
    if (my_rank == 0) {
        global_sum = local_sum; // Comenzar con el valor del raíz
        for (int i = 1; i < comm_sz; i++) {
            int temp_sum;
            MPI_Recv(&temp_sum, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            global_sum += temp_sum; // Agregar los valores restantes
        }
    } else {
        MPI_Send(&local_sum, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }

    // El proceso raíz contendrá la suma global
    if (my_rank == 0) {
        printf("Suma Global: %d\n", global_sum);
    }

    // Finalizar MPI
    MPI_Finalize();
    return 0;
}
