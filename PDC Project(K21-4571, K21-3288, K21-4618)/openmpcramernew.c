#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define X 3
#define Y (X + 1)

void displayMatrix(int mat[X][Y]) {
    for (int i = 0; i < X; i++) {
        for (int j = 0; j < Y; j++) {
            printf("%d\t | ", mat[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

void getCofactor(long long int mat[X][X], long long int temp[X][X], long long int p, long long int q, long long int n) {
    int i = 0, j = 0;
    for (int row = 0; row < n; row++) {
        for (int col = 0; col < n; col++) {
            if (row != p && col != q) {
                temp[i][j++] = mat[row][col];
                if (j == n - 1) {
                    j = 0;
                    i++;
                }
            }
        }
    }
}

long long int determinantOfMatrix(long long int mat[X][X], int n) {
    long long int D = 0;

    if (n == 1)
        return mat[0][0];

    long long int temp[X][X];

    int sign = 1;

    for (long long int f = 0; f < n; f++) {
        getCofactor(mat, temp, 0, f, n);
        D += sign * mat[0][f] * determinantOfMatrix(temp, n - 1);
        sign = -sign;
    }

    return D;
}

void findSolution(int coeff[X][Y]) {
    long long int d[X][X];
    for (int i = 0; i < X; i++) {
        for (int j = 0; j < X; j++) {
            d[i][j] = coeff[i][j];
        }
    }

    long long int dMat[X][X][X];
    for (int i = 0; i < X; i++) {
        for (int j = 0; j < X; j++) {
            for (int k = 0; k < X; k++) {
                if (k == i) {
                    dMat[i][j][k] = coeff[j][X];
                } else {
                    dMat[i][j][k] = coeff[j][k];
                }
            }
        }
    }

    long double _D = (long double)determinantOfMatrix(d, X);
    printf("det(A) : %Lf\n", _D);
    long double Dets[X];
    for (int i = 0; i < X; i++) {
        Dets[i] = (long double)determinantOfMatrix(dMat[i], X);
        printf("det(A%d) : %Lf\n", i + 1, Dets[i]);
    }

    if (_D != 0) {
        double xVals[X];
        for (int i = 0; i < X; i++) {
            xVals[i] = Dets[i] / _D;
            printf("x%d : %lf\n", i + 1, xVals[i]);
        }
    }
}

void findSolutionOMP(int coeff[X][Y], int threadCount) {
    omp_set_dynamic(0);
    omp_set_num_threads(threadCount);
    double startTime, endTime;
    long long int d[X][X];

    // Computation time for copying matrix coefficients
    startTime = omp_get_wtime();
    #pragma omp parallel for num_threads(threadCount) collapse(2)
    for (int i = 0; i < X; i++) {
        for (int j = 0; j < X; j++) {
            d[i][j] = coeff[i][j];
        }
    }
    endTime = omp_get_wtime();
    printf("Computation time (copying matrix coefficients): %lf seconds\n", endTime - startTime);

    // Communication time for copying matrix coefficients
    startTime = omp_get_wtime();
    #pragma omp parallel for num_threads(threadCount) collapse(2)
    for (int i = 0; i < X; i++) {
        for (int j = 0; j < X; j++) {
            // Do nothing, just iterate to measure communication time
        }
    }
    endTime = omp_get_wtime();
    printf("Communication time (copying matrix coefficients): %lf seconds\n", endTime - startTime);

    // Computation time for creating dMat matrices
    startTime = omp_get_wtime();
    long long int dMat[X][X][X];
    for (int i = 0; i < X; i++) {
        #pragma omp parallel for num_threads(threadCount) collapse(2)
        for (int j = 0; j < X; j++) {
            for (int k = 0; k < X; k++) {
                if (k == i) {
                    dMat[i][j][k] = coeff[j][X];
                } else {
                    dMat[i][j][k] = coeff[j][k];
                }
            }
        }
    }
    endTime = omp_get_wtime();
    printf("Computation time (creating dMat matrices): %lf seconds\n", endTime - startTime);

    // Communication time for creating dMat matrices
    startTime = omp_get_wtime();
    #pragma omp parallel for num_threads(threadCount) collapse(2)
    for (int i = 0; i < X; i++) {
        for (int j = 0; j < X; j++) {
            // Do nothing, just iterate to measure communication time
        }
    }
    endTime = omp_get_wtime();
    printf("Communication time (creating dMat matrices): %lf seconds\n", endTime - startTime);

    // Computation time for calculating determinant of A
    startTime = omp_get_wtime();
    long double _D = (long double)determinantOfMatrix(d, X);
    endTime = omp_get_wtime();
    printf("Computation time (determinant of A): %lf seconds\n", endTime - startTime);

    // Communication time for calculating determinant of A
    startTime = omp_get_wtime();
    #pragma omp parallel for num_threads(threadCount)
    for (int i = 0; i < X; i++) {
        // Do nothing, just iterate to measure communication time
    }
    endTime = omp_get_wtime();
    printf("Communication time (determinant of A): %lf seconds\n", endTime - startTime);

    // Computation time for calculating determinants of dMat matrices
    startTime = omp_get_wtime();
    long double Dets[X];
    #pragma omp parallel for num_threads(threadCount)
    for (int i = 0; i < X; i++) {
        Dets[i] = (long double)determinantOfMatrix(dMat[i], X);
    }
    endTime = omp_get_wtime();
    printf("Computation time (calculating determinants of dMat matrices): %lf seconds\n", endTime - startTime);

    // Communication time for calculating determinants of dMat matrices
    startTime = omp_get_wtime();
    #pragma omp parallel for num_threads(threadCount)
    for (int i = 0; i < X; i++) {
        // Do nothing, just iterate to measure communication time
    }
    endTime = omp_get_wtime();
    printf("Communication time (calculating determinants of dMat matrices): %lf seconds\n", endTime - startTime);

    // Computation and communication time for calculating x values
    if (_D != 0) {
        double xVals[X];
        startTime = omp_get_wtime();
        #pragma omp parallel for
        for (int i = 0; i < X; i++) {
            xVals[i] = Dets[i] / _D;
        }
        endTime = omp_get_wtime();
        printf("Computation and communication time (calculating x values): %lf seconds\n", endTime - startTime);
    }
}

int main() {
    int coeff[X][Y];
    srand(time(NULL));
    for (int i = 0; i < X; i++) {
        for (int j = 0; j < Y; j++) {
            coeff[i][j] = (rand() % 20) - 10;
        }
    }
    printf("Matrix\n");
    double time;


    time = omp_get_wtime();
    findSolutionOMP(coeff, 8);
    time = omp_get_wtime() - time;
    printf("time with 8 threads: %lf seconds\n\n", time);


    return 0;
}
