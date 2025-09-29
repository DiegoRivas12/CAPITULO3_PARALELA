#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

void print_vector(double* vector, int size) {
    for (int i = 0; i < size; i++) {
        printf("%f ", vector[i]);
    }
    printf("\n");
}

int main(int argc, char* argv[]) {
    int my_rank, comm_sz;
    int n; // Orden de la matriza
    double *A = NULL; // Matriz completa
    double *x = NULL; // Vector
    double *local_A; // Parte local de la matriz
    double *local_result; // Resultado local
    double *result; // Resultado global

    // Inicializar MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    if (my_rank == 0) {
        // Leer el tamaño de la matriz (n)
        printf("Enter the order of the matrix (n): ");
        scanf("%d", &n);

        // Asumimos que n es divisible por comm_sz
        A = malloc(n * n * sizeof(double));
        x = malloc(n * sizeof(double));
        result = malloc(n * sizeof(double));

        // Leer la matriz A
        printf("Enter the elements of the matrix (row-wise):\n");
        for (int i = 0; i < n * n; i++) {
            scanf("%lf", &A[i]);
        }

        // Leer el vector x
        printf("Enter the elements of the vector:\n");
        for (int i = 0; i < n; i++) {
            scanf("%lf", &x[i]);
        }
    }

    // Broadcast the order of the matrix
    MPI_Bcast(x, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Calcular el número de filas que cada proceso manejará
    int rows_per_process = n / comm_sz;

    // Asignar memoria para la parte local de la matriz y el resultado local
    local_A = (double*)malloc(rows_per_process * n * sizeof(double));
    local_result = (double*)malloc(rows_per_process * sizeof(double));

    // Distribuir las filas de la matriz A entre los procesos
    MPI_Scatter(A, rows_per_process * n, MPI_DOUBLE, local_A, rows_per_process * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Difundir el vector x entre todos los procesos
    //MPI_Bcast(x, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Inicializar el resultado local a cero
    for (int i = 0; i < rows_per_process; i++) {
        local_result[i] = 0.0;
    }

    // Realizar la multiplicación local
    for (int i = 0; i < rows_per_process; i++) {
        for (int j = 0; j < n; j++) {
            local_result[i] += local_A[i * n + j] * x[j];
        }
    }

    // Recolectar el resultado de todos los procesos en el proceso 0
    MPI_Reduce(local_result, result, rows_per_process, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // Imprimir el resultado final en el proceso 0
    if (my_rank == 0) {
        printf("Result of matrix-vector multiplication:\n");
        print_vector(result, n);
    }

    // Liberar memoria
    if (my_rank == 0) {
        free(A);
        free(x);
        free(result);
    }
    free(local_A);
    free(local_result);

    // Finalizar MPI
    MPI_Finalize();
    return 0;
}
