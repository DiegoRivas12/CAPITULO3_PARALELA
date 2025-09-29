#include <stdio.h>
#include <mpi.h>

int main(int argc, char* argv[]) {
    int my_rank, comm_sz;
    int local_sum = 1; // Cada proceso comienza con un valor locaal de 1
    int global_sum;

    // Inicializar MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    // Verificar si el tamaño del comunicador es una potencia de dos
    if (comm_sz & (comm_sz - 1)) {
        if (my_rank == 0) {
            printf("Error: El número de procesos debe ser una potencia de dos.\n");
        }
        MPI_Finalize();
        return 1;
    }

    // Realizar reducción estructurada en árbol
    int partner;
    for (int step = 1; step < comm_sz; step *= 2) {
        if (my_rank % (2 * step) == 0) { // Si my_rank es par
            partner = my_rank + step; // El compañero es el siguiente rango en el árbol
            MPI_Recv(&local_sum, 1, MPI_INT, partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            local_sum += local_sum; // Sumar el valor recibido
        } else {
            partner = my_rank - step; // El compañero es el rango anterior en el árbol
            MPI_Send(&local_sum, 1, MPI_INT, partner, 0, MPI_COMM_WORLD);
            break; // Salir una vez que se complete la operación de envío
        }
    }

    // El proceso raíz contendrá la suma global
    global_sum = local_sum;

    if (my_rank == 0) {
        printf("Suma Global: %d\n", global_sum);
    }

    // Finalizar MPI
    MPI_Finalize();
    return 0;
}
