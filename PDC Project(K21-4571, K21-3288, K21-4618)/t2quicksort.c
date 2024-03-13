#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define Max_Arr 5000 // size

void swap(int* a, int* b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

int pivot_place(int array[], int low, int high) {
    int pivot = array[high];
    int i = (low - 1);

    for (int j = low; j <= high - 1; j++) {
        if (array[j] <= pivot) {
            i++;
            swap(&array[i], &array[j]);
        }
    }

    swap(&array[i + 1], &array[high]);
    return (i + 1);
}

void quickSort(int array[], int low, int high) {
    if (low < high) {
        int pi = pivot_place(array, low, high);

        #pragma omp task
        quickSort(array, low, pi - 1);

        #pragma omp task
        quickSort(array, pi + 1, high);

        #pragma omp taskwait
    }
}

void display(int array[], int size) {
    for (int i = 0; i < size; i++)
        printf("%d ", array[i]);
    printf("\n");
}

int main() {
    double start_time, total_time;
    int array[Max_Arr];

    for (int i = 0; i < Max_Arr; i++) {
        array[i] = rand() % 2000 + 1;
    }

    int n = sizeof(array) / sizeof(array[0]);

    omp_set_num_threads(8); // setting the number of threads
    start_time = omp_get_wtime();

    #pragma omp parallel
    {
        #pragma omp single nowait
        quickSort(array, 0, n - 1);
    }

    total_time = omp_get_wtime() - start_time;
    printf("\nExecution time was %lf seconds\n", total_time);

    // Display sorted array
    printf("Sorted array:\n");
    display(array, Max_Arr);

    return 0;
}
