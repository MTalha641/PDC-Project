#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

//////////////////////////////////////////////////////////
void swap(int* array, int i, int j){
    int temp = array[i];
    array[i] = array[j];
    array[j] = temp;
}

//////////////////////////////////////////////////////////
void quicksort(int* array, int start, int end){
    int pivot, index;
    if (end <= 1) {
        return;
    }
    pivot = array[start + end / 2];
    swap(array, start, start + end / 2);
    index = start;
    for (int i = start + 1; i < start + end; i++){
        if (array[i] < pivot) {
            index++;
            swap(array, i, index);
        }
    }
    swap(array, start, index);
    quicksort(array, start, index - start);
    quicksort(array, index + 1, start + end - index - 1);
}

//////////////////////////////////////////////////////////
int* merge(int* array1, int num1, int* array2, int num2){
    int* result = (int*)malloc((num1 + num2) * sizeof(int));
    int i = 0, j = 0, k;

    for (k = 0; k < num1 + num2; k++) {
        if (i >= num1) {
            result[k] = array2[j];
            j++;
        }
        else if (j >= num2) {
            result[k] = array1[i];
            i++;
        }
        else if (array1[i] < array2[j]) {
            result[k] = array1[i];
            i++;
        }
        else {
            result[k] = array2[j];
            j++;
        }
    }
    return result;
}

//////////////////////////////////////////////////////////
int main(int argc, char* argv[]){
    int elements_number = 5000; // no of sizes
    int* data = NULL;
    int chunk_size, own_chunk_size;
    int* chunk;
    double time_taken, communication_time, computation_time;

    MPI_Status status;

    int number_of_process, process_rank;
    int rc = MPI_Init(&argc, &argv);

    if (rc != MPI_SUCCESS){
        printf("Error in creating MPI program.\nTerminating......\n");
        MPI_Abort(MPI_COMM_WORLD, rc);
    }

    MPI_Comm_size(MPI_COMM_WORLD, &number_of_process);
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);

    if (process_rank == 0) {
        printf("Number of Elements is %d \n", elements_number);

        chunk_size = (elements_number % number_of_process == 0) ?
            (elements_number / number_of_process) : (elements_number / (number_of_process - 1));

        data = (int *)malloc(number_of_process * chunk_size * sizeof(int));

        printf("Reading the array...\n");
        for (int i = 0; i < elements_number; i++){
            data[i] = rand() % 1000;
        }

        for (int i = elements_number; i < number_of_process * chunk_size; i++){
            data[i] = 0;
        }

        printf("\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);

    time_taken = MPI_Wtime();

    MPI_Bcast(&elements_number, 1, MPI_INT, 0, MPI_COMM_WORLD);

    chunk_size = (elements_number % number_of_process == 0) ? (elements_number / number_of_process) : (elements_number / (number_of_process - 1));

    chunk = (int *)malloc(chunk_size * sizeof(int));

    MPI_Scatter(data, chunk_size, MPI_INT, chunk, chunk_size, MPI_INT, 0, MPI_COMM_WORLD);

    free(data);
    data = NULL;

    own_chunk_size = (elements_number >= chunk_size * (process_rank + 1)) ? chunk_size : (elements_number - chunk_size * process_rank);

    time_taken = MPI_Wtime() - time_taken;
    communication_time = time_taken; // Communication time for scatter

    time_taken = MPI_Wtime();

    quicksort(chunk, 0, own_chunk_size);

    for (int step = 1; step < number_of_process; step = 2 * step){
        if (process_rank % (2 * step) != 0) {
            MPI_Send(chunk, own_chunk_size, MPI_INT,
                process_rank - step, 0,
                MPI_COMM_WORLD);
            break;
        }

        if (process_rank + step < number_of_process) {
            int received_chunk_size
                = (elements_number
                    >= chunk_size
                    * (process_rank + 2 * step))
                ? (chunk_size * step)
                : (elements_number
                    - chunk_size
                    * (process_rank + step));

            int* chunk_received;
            chunk_received = (int*)malloc(received_chunk_size * sizeof(int));
            MPI_Recv(chunk_received, received_chunk_size, MPI_INT, process_rank + step, 0,
                MPI_COMM_WORLD, &status);

            data = merge(chunk, own_chunk_size, chunk_received, received_chunk_size);

            free(chunk);
            free(chunk_received);
            chunk = data;
            own_chunk_size = own_chunk_size + received_chunk_size;
        }
    }

    time_taken = MPI_Wtime() - time_taken;
	 // Communication time for merge

    if (process_rank == 0){
        computation_time = time_taken; // Computation time for quicksort

        printf("Total number of Elements given as input : %d\n", elements_number);
        printf("Sorted arrayay is: \n");
        for (int i = 0; i < elements_number; i++) {
            printf("%d ", chunk[i]);
        }
        printf("\n\nQuicksort %d ints on %d procs: %f secs\n", elements_number, number_of_process, computation_time);
        printf("Communication time: %f secs\n", communication_time);
    }

    MPI_Finalize();
    return 0;
}

