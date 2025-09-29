#include <stdio.h>
#include <stdlib.h>

void print_vector(double* vector, int size) {
    for (int i = 0; i < size; i++) {
        printf("%f ", vector[i]);
    }
    printf("\n");
}

int main() {
    int n; // Orden de la matrizz
    double *A; // Matriz
    double *x; // Vector
    double *result; // Resultado

    // Leer el tamaño de la matriz (n)
    printf("Enter the order of the matrix (n): ");
    scanf("%d", &n);

    // Asumimos que la matriz es cuadrada de orden n
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

    // Inicializar el resultado a cero
    for (int i = 0; i < n; i++) {
        result[i] = 0.0;
    }

    // Realizar la multiplicación
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            result[i] += A[i * n + j] * x[j];
        }
    }

    // Imprimir el resultado
    printf("Result of matrix-vector multiplication:\n");
    print_vector(result, n);

    // Liberar memoria
    free(A);
    free(x);
    free(result);

    return 0;
}
