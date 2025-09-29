#include <stdio.h>
#include <mpi.h>
double f(double x) {
    // Definición de la función a integrar
    return x * x; // Ejemplo: f(x) = x^2
}
double Trap(
    double left_endpt,  /* in */
    double right_endpt, /* in */
    int trap_count,     /* in */
    double base_len     /* in */a
) {
    double estimate, x;
    int i;

    estimate = (f(left_endpt) + f(right_endpt)) / 2.0; // Cálculo inicial
    for (i = 1; i <= trap_count - 1; i++) { // Bucle para sumar las áreas de los trapecios
        x = left_endpt + i * base_len;
        estimate += f(x);
    }
    estimate = estimate * base_len; // Multiplicar por la longitud de la base

    return estimate; // Devolver la estimación de la integral
} /* Trap */




int main(void) {
    int my_rank, comm_sz, n = 1024, local_n;
    double a = 0.0, b = 3.0, h, local_a, local_b;
    double local_int, total_int;
    int source;

    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    h = (b - a) / n; // h es el mismo para todos los procesos
    local_n = n / comm_sz; // También lo es el número de trapecios

    local_a = a + my_rank * local_n * h;
    local_b = local_a + local_n * h;
    local_int = Trap(local_a, local_b, local_n, h);

    if (my_rank != 0) {
        MPI_Send(&local_int, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    } else {
        total_int = local_int;
        for (source = 1; source < comm_sz; source++) {
            MPI_Recv(&local_int, 1, MPI_DOUBLE, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            total_int += local_int;
        }
    }

    if (my_rank == 0) {
        printf("With n = %d trapezoids, our estimate\n", n);
        printf("of the integral from %f to %f = %.15e\n", a, b, total_int);
    }

    MPI_Finalize();
    return 0;
}


