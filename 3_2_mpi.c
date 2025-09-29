#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Función para realizar los lanzamientos de dardoss
long long toss_darts(long long tosses) {
    long long count = 0;
    double x, y;

    // Semilla basada en el tiempo y el rango del proceso para asegurar aleatoriedad
    srand(time(NULL) + rand()); // Cada proceso tiene su propia semilla

    for (long long toss = 0; toss < tosses; toss++) {
        x = (double)rand() / RAND_MAX * 2.0 - 1.0; // Número aleatorio en [-1, 1]
        y = (double)rand() / RAND_MAX * 2.0 - 1.0; // Número aleatorio en [-1, 1]
        if (x * x + y * y <= 1.0) {
            count++;
        }
    }

    return count;
}

int main(int argc, char** argv) {
    int rank, size;
    long long number_of_tosses;
    long long local_count = 0;
    long long total_count = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Solo el proceso 0 lee el número total de lanzamientos
    if (rank == 0) {
        printf("Enter the number of tosses: ");
        scanf("%lld", &number_of_tosses);
    }

    // Difundir el número total de lanzamientos a todos los procesos
    MPI_Bcast(&number_of_tosses, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    // Cada proceso calcula cuántos lanzamientos le corresponden
    long long tosses_per_process = number_of_tosses / size;

    // Llamar a la función para lanzar dardos y contar los que caen en el círculo
    local_count = toss_darts(tosses_per_process);

    // Reducir los conteos locales al proceso 0
    MPI_Reduce(&local_count, &total_count, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    // Solo el proceso 0 imprime el resultado
    if (rank == 0) {
        double pi_estimate = 4.0 * total_count / number_of_tosses;
        printf("Estimated Pi = %.6f\n", pi_estimate);
    }

    MPI_Finalize();
    return 0;
}
