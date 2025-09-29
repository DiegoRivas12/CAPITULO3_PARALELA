#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

long long toss_darts(long long number_of_tosses) {
    long long count = 0;
    for (long long i = 0; i < number_of_tosses; i++) {
        double x = (double)rand() / RAND_MAX * 2 - 1; // Número entre -1 y 1a
        double y = (double)rand() / RAND_MAX * 2 - 1; // Número entre -1 y 1
        if (x * x + y * y <= 1) {
            count++;
        }
    }
    return count;
}

int main(int argc, char *argv[]) {
    int my_rank, comm_sz;
    long long number_of_tosses, local_tosses, local_count;
    long long total_count = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    if (my_rank == 0) {
        printf("Enter the number of tosses: ");
        scanf("%lld", &number_of_tosses);
    }

    // Dividir el total de lanzamientos entre los procesos
    MPI_Bcast(&number_of_tosses, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    local_tosses = number_of_tosses / comm_sz; // Cada proceso lanza su parte

    // Semilla para la generación aleatoria
    srand(time(NULL) + my_rank); // Semilla diferente para cada proceso

    // Cada proceso lanza dardos
    local_count = toss_darts(local_tosses);

    // Proceso 0 recibe los conteos de cada proceso
    if (my_rank == 0) {
        total_count = local_count; // Comienza con el conteo del proceso 0
        for (int source = 1; source < comm_sz; source++) {
            long long temp_count;
            MPI_Recv(&temp_count, 1, MPI_LONG_LONG, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            total_count += temp_count; // Sumar el conteo recibido
        }
    } else {
        // Enviar el conteo local al proceso 0
        MPI_Send(&local_count, 1, MPI_LONG_LONG, 0, 0, MPI_COMM_WORLD);
    }

    // El proceso 0 calcula e imprime la estimación de π
    if (my_rank == 0) {
        double pi_estimate = 4.0 * total_count / number_of_tosses;
        printf("Estimated Pi = %.6f\n", pi_estimate);
    }

    MPI_Finalize();
    return 0;
}
