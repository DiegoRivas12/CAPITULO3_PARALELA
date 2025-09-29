#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int numeroProcesadores, id_Proceso;

    float *A, // Matriz global a multiplicar
          *x, // Vector a multiplicar
          *y, // Vector resultadoa
          *local_A,  // Matriz local de cada proceso
          *local_y;  // Porción local del resultado en cada proceso

    double tInicio, // Tiempo en el que comienza la ejecución
           Tpar, Tseq;   

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numeroProcesadores);
    MPI_Comm_rank(MPI_COMM_WORLD, &id_Proceso);

    int n;

    if (argc <= 1) { // si no se pasa el tamaño de la matriz, se elige n=10
        if (id_Proceso == 0)
            printf("The dimension N of the matrix is missing (N x N matrix)\n");
        MPI_Finalize();
        return 0;
    } else {
        n = atoi(argv[1]);
    }

    x = (float*)malloc(n * sizeof(float)); // Reservamos espacio para el vector x (n floats).

    // Proceso 0 genera matriz A y vector x
    if (id_Proceso == 0) {
        A = (float*)malloc(n * n * sizeof(float)); // Reservamos espacio para la matriz (n x n floats)
        y = (float*)malloc(n * sizeof(float)); // Reservamos espacio para el vector resultado final y (n floats)

        // Rellena la matriz y el vector
        for (int i = 0; i < n; i++) {
            x[i] = (float)(1.5 * (1 + (5 * i) % 3) / (1 + (i) % 5));
            for (int j = 0; j < n; j++) {
                A[i * n + j] = (float)(1.5 * (1 + (5 * (i + j)) % 3) / (1 + (i + j) % 5));
            }
        }

        // Imprimir la matriz A
        printf("Matriz A:\n");
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                printf("%f ", A[i * n + j]);
            }
            printf("\n");
        }

        // Imprimir el vector x
        printf("Vector x:\n");
        for (int i = 0; i < n; i++) {
            printf("%f ", x[i]);
        }
        printf("\n");
    }

    // Cada proceso reserva espacio para su porción de A y para el vector x 
    const int local_A_size = n * n / numeroProcesadores;
    const int local_y_size = n / numeroProcesadores;
    local_A = (float*)malloc(local_A_size * sizeof(float)); // Reservamos espacio para la matriz local
    local_y = (float*)malloc(local_y_size * sizeof(float)); // Reservamos espacio para el vector local y

    // Repartimos un bloque de filas de A a cada proceso
    MPI_Scatter(A, // Matriz que vamos a compartir
                 local_A_size, // Número de elementos a entregar
                 MPI_FLOAT, // Tipo de dato a enviar
                 local_A, // Vector en el que almacenar los datos
                 local_A_size, // Número de elementos a recibir
                 MPI_FLOAT, // Tipo de dato a recibir
                 0, // Proceso raíz que envía los datos
                 MPI_COMM_WORLD); // Comunicador utilizado (En este caso, el global)

    // Difundimos el vector x entre todos los procesos
    MPI_Bcast(x, // Dato a compartir
              n, // Número de elementos que se van a enviar y recibir
              MPI_FLOAT, // Tipo de dato que se compartirá
              0, // Proceso raíz que envía los datos
              MPI_COMM_WORLD); // Comunicador utilizado (En este caso, el global)

    // Hacemos una barrera para asegurar que todos los procesos comiencen la ejecución
    // a la vez, para tener mejor control del tiempo empleado
    MPI_Barrier(MPI_COMM_WORLD);
    // Inicio de medición de tiempo
    tInicio = MPI_Wtime();

    for (int i = 0; i < local_y_size; i++) {
        local_y[i] = 0.0;
        for (int j = 0; j < n; j++) {
            local_y[i] += local_A[i * n + j] * x[j];
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    // Fin de medición de tiempo
    Tpar = MPI_Wtime() - tInicio;

    // Recogemos los datos de la multiplicación, cada proceso envía su resultado
    MPI_Gather(local_y, // Dato que envía cada proceso
               local_y_size, // Número de elementos que se envían
               MPI_FLOAT, // Tipo del dato que se envía
               y, // Vector en el que se recolectan los datos
               local_y_size, // Número de datos que se esperan recibir por cada proceso
               MPI_FLOAT, // Tipo del dato que se recibirá
               0, // Proceso que va a recibir los datos
               MPI_COMM_WORLD); // Canal de comunicación (Comunicador Global)

    // Terminamos la ejecución de los procesos, después de esto solo existirá
    // el proceso 0
    MPI_Finalize();

    if (id_Proceso == 0) {
        float *comprueba = (float *)malloc(n * sizeof(float));
        // Calculamos la multiplicación secuencial para 
        // después comprobar que es correcta la solución.

        tInicio = MPI_Wtime();
        for (int i = 0; i < n; i++) {
            comprueba[i] = 0;
            for (int j = 0; j < n; j++) {
                comprueba[i] += A[i * n + j] * x[j];
            }
        }
        Tseq = MPI_Wtime() - tInicio;

        int errores = 0;
        for (unsigned int i = 0; i < n; i++) {   
            printf("\t%f\t|\t%f\n", y[i], comprueba[i]);
            if (comprueba[i] != y[i])
                errores++;
        }
        printf(".......Obtained and expected result can be seen above.......\n");

        free(y);
        free(comprueba);
        free(A);

        if (errores) {
            printf("Found %d Errors!!!\n", errores);
        } else {
            printf("No Errors!\n\n");
            printf("...Parallel time (without initial distribution and final gathering)= %f seconds.\n\n", Tpar);
            printf("...Sequential time= %f seconds.\n\n", Tseq);
        }
    }

    free(local_A);
    free(local_y);
    free(x);

    return 0;
}
