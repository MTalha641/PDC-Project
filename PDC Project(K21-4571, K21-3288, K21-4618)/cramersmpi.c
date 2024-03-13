#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define MATRIX_SIZE 9

void printMatrix(double matrix[MATRIX_SIZE][MATRIX_SIZE]) {
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            printf("%f\t", matrix[i][j]);
        }
        printf("\n");
    }
}

double calculateDeterminant(double matrix[MATRIX_SIZE][MATRIX_SIZE]) {
    return matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[2][1] * matrix[1][2]) -
           matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[2][0] * matrix[1][2]) +
           matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[2][0] * matrix[1][1]);
}

void replaceColumn(double matrix[MATRIX_SIZE][MATRIX_SIZE], double column[MATRIX_SIZE], int col) {
    for (int i = 0; i < MATRIX_SIZE; i++) {
        matrix[i][col] = column[i];
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    double matrix[MATRIX_SIZE][MATRIX_SIZE] = {
        {2.0, -1.0, 1.0,2,5,2,1,5,6},
        {-3.0, -1.0, 2.0,6,8,2,2,5,1},
        {-2.0, 1.0, 2.0,2,6,3,3,6,7}
    };

    double column[MATRIX_SIZE] = {8.0, -11.0, -3.0};
    double determinant;

    double start_time, end_time;  // Add timing variables

    if (world_rank == 0) {
        start_time = MPI_Wtime();  // Start timing for computation

        printf("Original Coefficient Matrix:\n");
        printMatrix(matrix);
        printf("\n");

        printf("Right-hand Side Column:\n");
        for (int i = 0; i < MATRIX_SIZE; i++) {
            printf("%f\n", column[i]);
        }
        printf("\n");

        determinant = calculateDeterminant(matrix);
        printf("Determinant of Coefficient Matrix: %f\n", determinant);

        end_time = MPI_Wtime();  // End timing for computation
        printf("Computation Time: %f seconds\n", end_time - start_time);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (world_rank == 0) {
        start_time = MPI_Wtime();  // Start timing for communication
    }

    MPI_Bcast(&determinant, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        end_time = MPI_Wtime();  // End timing for communication
        printf("Communication Time: %f seconds\n", end_time - start_time);
    }

    int col;
    for (col = world_rank; col < MATRIX_SIZE + world_rank; col += world_size) {
        double temp_matrix[MATRIX_SIZE][MATRIX_SIZE];
        for (int i = 0; i < MATRIX_SIZE; i++) {
            for (int j = 0; j < MATRIX_SIZE; j++) {
                temp_matrix[i][j] = matrix[i][j];
            }
        }

        replaceColumn(temp_matrix, column, col % MATRIX_SIZE);

        double result = calculateDeterminant(temp_matrix) / determinant;

        if (world_rank == 0) {
            printf("Solution for X%d: %f\n", col % MATRIX_SIZE + 1, result);
        }
    }

    MPI_Finalize();

    return 0;
}

