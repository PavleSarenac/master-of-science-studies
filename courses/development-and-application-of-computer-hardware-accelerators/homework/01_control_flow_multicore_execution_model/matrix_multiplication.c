#define _POSIX_C_SOURCE 199309L
#define NUMBER_OF_NANOSECONDS_IN_ONE_SECOND 1e9
#define ONE_BILLION_FLOATING_POINT_OPERATIONS 1e9
#define TOTAL_NUMBER_OF_PHYSICAL_PROCESSOR_CORES 4

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <omp.h>

static double get_current_number_of_seconds() 
{
    struct timespec time_specification;
    clock_gettime(CLOCK_MONOTONIC, &time_specification);
    return time_specification.tv_sec + time_specification.tv_nsec / NUMBER_OF_NANOSECONDS_IN_ONE_SECOND;
}

static void* safe_malloc(size_t size_in_bytes)
{
    void *pointer = malloc(size_in_bytes);
    if (!pointer)
    {
        fprintf(stderr, "Memory allocation failed.\n");
        exit(1);
    }
    return pointer;
}

static void free_matrix_as_2d_array(double **matrix, int number_of_rows)
{
    for (int current_row_index = 0; current_row_index < number_of_rows; current_row_index++)
    {
        free(matrix[current_row_index]);
    }
    free(matrix);
}

static double **allocate_matrix_as_2d_array(int number_of_rows, int number_of_columns)
{
    size_t single_matrix_row_pointer_size_in_bytes = sizeof(double *);
    size_t all_matrix_row_pointers_size_in_bytes = (size_t)number_of_rows * single_matrix_row_pointer_size_in_bytes;

    double **matrix = (double **)safe_malloc(all_matrix_row_pointers_size_in_bytes);

    size_t single_matrix_element_size_in_bytes = sizeof(double);
    size_t single_matrix_row_size_in_bytes = number_of_columns * single_matrix_element_size_in_bytes;

    for (int current_row_index = 0; current_row_index < number_of_rows; current_row_index++)
    {
        matrix[current_row_index] = (double *)safe_malloc(single_matrix_row_size_in_bytes);
    }

    return matrix;
}

static double *allocate_matrix_as_1d_array(int number_of_rows, int number_of_columns)
{
    size_t total_number_of_matrix_elements = (size_t)number_of_rows * number_of_columns;
    size_t single_matrix_element_size_in_bytes = sizeof(double);
    size_t matrix_size_in_bytes = total_number_of_matrix_elements * single_matrix_element_size_in_bytes;

    double *matrix = (double *)safe_malloc(matrix_size_in_bytes);
    return matrix;
}

static void populate_matrix_as_2d_array_with_random_values(double **matrix, int number_of_rows, int number_of_columns)
{
    for (int current_row_index = 0; current_row_index < number_of_rows; current_row_index++)
    {
        for (int current_column_index = 0; current_column_index < number_of_columns; current_column_index++)
        {
            matrix[current_row_index][current_column_index] = (double)rand() / RAND_MAX;
        }
    }
}

static void populate_matrix_as_1d_array_with_random_values(double *matrix, int number_of_rows, int number_of_columns)
{
    unsigned long total_number_of_matrix_elements = (unsigned long)number_of_rows * number_of_columns;
    for (int current_matrix_element_index = 0; current_matrix_element_index < total_number_of_matrix_elements; current_matrix_element_index++)
    {
        matrix[current_matrix_element_index] = (double)rand() / RAND_MAX;
    }
}

static void populate_matrix_as_2d_array_with_zeros(double **matrix, int number_of_rows, int number_of_columns)
{
    size_t single_matrix_row_size_in_bytes = (size_t)number_of_columns * sizeof(double);
    for (int current_row_index = 0; current_row_index < number_of_rows; current_row_index++)
    {
        memset(matrix[current_row_index], 0, single_matrix_row_size_in_bytes);
    }
}

static void populate_matrix_as_1d_array_with_zeros(double *matrix, int number_of_rows, int number_of_columns)
{
    size_t total_number_of_matrix_elements = (size_t)number_of_rows * number_of_columns;
    size_t single_matrix_element_size_in_bytes = sizeof(double);
    size_t matrix_size_in_bytes = total_number_of_matrix_elements * single_matrix_element_size_in_bytes;

    memset(matrix, 0, matrix_size_in_bytes);
}

static double **transpose_matrix_as_2d_array(double **matrix, int number_of_rows, int number_of_columns)
{
    double **transposed_matrix = allocate_matrix_as_2d_array(number_of_rows = number_of_columns, number_of_columns = number_of_rows);
    for (int current_row_index = 0; current_row_index < number_of_rows; current_row_index++)
    {
        for (int current_column_index = 0; current_column_index < number_of_columns; current_column_index++)
        {
            transposed_matrix[current_column_index][current_row_index] = matrix[current_row_index][current_column_index];
        }
    }
    return transposed_matrix;
}

static double *transpose_matrix_as_1d_array(double *matrix, int number_of_rows, int number_of_columns)
{
    double *transposed_matrix = allocate_matrix_as_1d_array(number_of_rows = number_of_columns, number_of_columns = number_of_rows);
    for (int current_row_index = 0; current_row_index < number_of_rows; current_row_index++)
    {
        for (int current_column_index = 0; current_column_index < number_of_columns; current_column_index++)
        {
            transposed_matrix[current_column_index * number_of_rows + current_row_index] = matrix[current_row_index * number_of_columns + current_column_index];
        }
    }
    return transposed_matrix;
}

void matrix_multiplication_sequential_01(double **matrix_1, double **matrix_2, double **matrix_3, int number_of_rows, int number_of_columns)
{
    for (int matrix_2_column_index = 0; matrix_2_column_index < number_of_columns; matrix_2_column_index++)
    {
        for (int matrix_1_row_index = 0; matrix_1_row_index < number_of_rows; matrix_1_row_index++)
        {
            double current_dot_product = 0.0;
            for (int matrix_1_column_index = 0; matrix_1_column_index < number_of_columns; matrix_1_column_index++)
            {
                current_dot_product += matrix_1[matrix_1_row_index][matrix_1_column_index] * matrix_2[matrix_1_column_index][matrix_2_column_index];
            }
            matrix_3[matrix_1_row_index][matrix_2_column_index] = current_dot_product;
        }
    }
}

void matrix_multiplication_sequential_02(double **matrix_1, double **matrix_2, double **matrix_3, int number_of_rows, int number_of_columns)
{
    double **transposed_matrix_2 = transpose_matrix_as_2d_array(matrix_2, number_of_rows, number_of_columns);

    for (int matrix_1_row_index = 0; matrix_1_row_index < number_of_rows; matrix_1_row_index++)
    {
        for (int transposed_matrix_2_row_index = 0; transposed_matrix_2_row_index < number_of_columns; transposed_matrix_2_row_index++)
        {
            double current_dot_product = 0.0;
            for (int matrix_1_column_index = 0; matrix_1_column_index < number_of_columns; matrix_1_column_index++)
            {
                current_dot_product += matrix_1[matrix_1_row_index][matrix_1_column_index] * transposed_matrix_2[transposed_matrix_2_row_index][matrix_1_column_index];
            }
            matrix_3[matrix_1_row_index][transposed_matrix_2_row_index] = current_dot_product;
        }
    }

    free_matrix_as_2d_array(transposed_matrix_2, number_of_rows = number_of_columns);
}

void matrix_multiplication_sequential_03(double *matrix_1, double *matrix_2, double *matrix_3, int number_of_rows, int number_of_columns)
{
    double *transposed_matrix_2 = transpose_matrix_as_1d_array(matrix_2, number_of_rows, number_of_columns);

    for (int matrix_1_row_index = 0; matrix_1_row_index < number_of_rows; matrix_1_row_index++)
    {
        for (int transposed_matrix_2_row_index = 0; transposed_matrix_2_row_index < number_of_columns; transposed_matrix_2_row_index++)
        {
            double *matrix_1_current_row = matrix_1 + matrix_1_row_index * number_of_columns;
            double *transposed_matrix_2_current_row = transposed_matrix_2 + transposed_matrix_2_row_index * number_of_rows;
            double current_dot_product = 0.0;
            for (int matrix_1_column_index = 0; matrix_1_column_index < number_of_columns; matrix_1_column_index++)
            {
                current_dot_product += matrix_1_current_row[matrix_1_column_index] * transposed_matrix_2_current_row[matrix_1_column_index];
            }
            matrix_3[matrix_1_row_index * number_of_columns + transposed_matrix_2_row_index] = current_dot_product;
        }
    }

    free(transposed_matrix_2);
}

void matrix_multiplication_parallel_03(double *matrix_1, double *matrix_2, double *matrix_3, int number_of_rows, int number_of_columns)
{
    double *transposed_matrix_2 = transpose_matrix_as_1d_array(matrix_2, number_of_rows, number_of_columns);

    #pragma omp parallel for schedule(static)
    for (int matrix_1_row_index = 0; matrix_1_row_index < number_of_rows; matrix_1_row_index++)
    {
        for (int transposed_matrix_2_row_index = 0; transposed_matrix_2_row_index < number_of_columns; transposed_matrix_2_row_index++)
        {
            double *matrix_1_current_row = matrix_1 + matrix_1_row_index * number_of_columns;
            double *transposed_matrix_2_current_row = transposed_matrix_2 + transposed_matrix_2_row_index * number_of_rows;
            double current_dot_product = 0.0;
            for (int matrix_1_column_index = 0; matrix_1_column_index < number_of_columns; matrix_1_column_index++)
            {
                current_dot_product += matrix_1_current_row[matrix_1_column_index] * transposed_matrix_2_current_row[matrix_1_column_index];
            }
            matrix_3[matrix_1_row_index * number_of_columns + transposed_matrix_2_row_index] = current_dot_product;
        }
    }

    free(transposed_matrix_2);
}

void benchmark_sequential_01(int number_of_runs, int number_of_rows, int number_of_columns)
{
    char benchmarking_result_file_name[128];
    snprintf(benchmarking_result_file_name, sizeof(benchmarking_result_file_name), "benchmarking_results/sequential/sequential_01/sequential_01_%d_%d.csv", number_of_rows, number_of_columns);

    FILE *benchmarking_result_file = fopen(benchmarking_result_file_name, "w");
    if (!benchmarking_result_file)
    {
        fprintf(stderr, "Bencmarking result file opening failed.\n");
        exit(1);
    }

    printf("Benchmarking of sequential implementation number 01 started.\n");
    fprintf(benchmarking_result_file, "run_index,number_of_matrix_rows,number_of_matrix_columns,run_duration_in_seconds,gflops\n");

    for (int current_run_index = 0; current_run_index < number_of_runs; current_run_index++)
    {
        double **matrix_1 = allocate_matrix_as_2d_array(number_of_rows, number_of_columns);
        double **matrix_2 = allocate_matrix_as_2d_array(number_of_rows, number_of_columns);
        double **matrix_3 = allocate_matrix_as_2d_array(number_of_rows, number_of_columns);

        populate_matrix_as_2d_array_with_random_values(matrix_1, number_of_rows, number_of_columns);
        populate_matrix_as_2d_array_with_random_values(matrix_2, number_of_rows, number_of_columns);
        populate_matrix_as_2d_array_with_zeros(matrix_3, number_of_rows, number_of_columns);

        double start_time_in_seconds = get_current_number_of_seconds();
        matrix_multiplication_sequential_01(matrix_1, matrix_2, matrix_3, number_of_rows, number_of_columns);
        double end_time_in_seconds = get_current_number_of_seconds();

        double run_duration_in_seconds = end_time_in_seconds - start_time_in_seconds;

        unsigned long long total_number_of_multiplications = (unsigned long long)number_of_rows * number_of_rows * number_of_rows;
        unsigned long long total_number_of_additions = (unsigned long long)number_of_rows * number_of_rows * (number_of_rows - 1);
        unsigned long long total_number_of_floating_point_operations = total_number_of_multiplications + total_number_of_additions;

        double total_number_of_floating_point_operations_per_second = (double)total_number_of_floating_point_operations / run_duration_in_seconds;
        double total_number_of_billions_of_floating_point_operations_per_second = total_number_of_floating_point_operations_per_second / ONE_BILLION_FLOATING_POINT_OPERATIONS;

        fprintf(benchmarking_result_file, "%d,%d,%d,%.5f,%.5f\n", current_run_index + 1, number_of_rows, number_of_columns, run_duration_in_seconds, total_number_of_billions_of_floating_point_operations_per_second);

        free_matrix_as_2d_array(matrix_1, number_of_rows);
        free_matrix_as_2d_array(matrix_2, number_of_rows);
        free_matrix_as_2d_array(matrix_3, number_of_rows);

        printf("Run %d complete.\n", current_run_index + 1);
    }
    printf("Benchmarking of sequential implementation number 01 ended.\n");

    fclose(benchmarking_result_file);
}

void benchmark_sequential_02(int number_of_runs, int number_of_rows, int number_of_columns)
{
    char benchmarking_result_file_name[128];
    snprintf(benchmarking_result_file_name, sizeof(benchmarking_result_file_name), "benchmarking_results/sequential/sequential_02/sequential_02_%d_%d.csv", number_of_rows, number_of_columns);

    FILE *benchmarking_result_file = fopen(benchmarking_result_file_name, "w");
    if (!benchmarking_result_file)
    {
        fprintf(stderr, "Bencmarking result file opening failed.\n");
        exit(1);
    }

    printf("Benchmarking of sequential implementation number 02 started.\n");
    fprintf(benchmarking_result_file, "run_index,number_of_matrix_rows,number_of_matrix_columns,run_duration_in_seconds,gflops\n");

    for (int current_run_index = 0; current_run_index < number_of_runs; current_run_index++)
    {
        double **matrix_1 = allocate_matrix_as_2d_array(number_of_rows, number_of_columns);
        double **matrix_2 = allocate_matrix_as_2d_array(number_of_rows, number_of_columns);
        double **matrix_3 = allocate_matrix_as_2d_array(number_of_rows, number_of_columns);

        populate_matrix_as_2d_array_with_random_values(matrix_1, number_of_rows, number_of_columns);
        populate_matrix_as_2d_array_with_random_values(matrix_2, number_of_rows, number_of_columns);
        populate_matrix_as_2d_array_with_zeros(matrix_3, number_of_rows, number_of_columns);

        double start_time_in_seconds = get_current_number_of_seconds();
        matrix_multiplication_sequential_02(matrix_1, matrix_2, matrix_3, number_of_rows, number_of_columns);
        double end_time_in_seconds = get_current_number_of_seconds();

        double run_duration_in_seconds = end_time_in_seconds - start_time_in_seconds;

        unsigned long long total_number_of_multiplications = (unsigned long long)number_of_rows * number_of_rows * number_of_rows;
        unsigned long long total_number_of_additions = (unsigned long long)number_of_rows * number_of_rows * (number_of_rows - 1);
        unsigned long long total_number_of_floating_point_operations = total_number_of_multiplications + total_number_of_additions;

        double total_number_of_floating_point_operations_per_second = (double)total_number_of_floating_point_operations / run_duration_in_seconds;
        double total_number_of_billions_of_floating_point_operations_per_second = total_number_of_floating_point_operations_per_second / ONE_BILLION_FLOATING_POINT_OPERATIONS;

        fprintf(benchmarking_result_file, "%d,%d,%d,%.5f,%.5f\n", current_run_index + 1, number_of_rows, number_of_columns, run_duration_in_seconds, total_number_of_billions_of_floating_point_operations_per_second);

        free_matrix_as_2d_array(matrix_1, number_of_rows);
        free_matrix_as_2d_array(matrix_2, number_of_rows);
        free_matrix_as_2d_array(matrix_3, number_of_rows);

        printf("Run %d complete.\n", current_run_index + 1);
    }
    printf("Benchmarking of sequential implementation number 02 ended.\n");
    
    fclose(benchmarking_result_file);
}

void benchmark_sequential_03(int number_of_runs, int number_of_rows, int number_of_columns)
{
    char benchmarking_result_file_name[128];
    snprintf(benchmarking_result_file_name, sizeof(benchmarking_result_file_name), "benchmarking_results/sequential/sequential_03/sequential_03_%d_%d.csv", number_of_rows, number_of_columns);

    FILE *benchmarking_result_file = fopen(benchmarking_result_file_name, "w");
    if (!benchmarking_result_file)
    {
        fprintf(stderr, "Bencmarking result file opening failed.\n");
        exit(1);
    }

    printf("Benchmarking of sequential implementation number 03 started.\n");
    fprintf(benchmarking_result_file, "run_index,number_of_matrix_rows,number_of_matrix_columns,run_duration_in_seconds,gflops\n");

    for (int current_run_index = 0; current_run_index < number_of_runs; current_run_index++)
    {
        double *matrix_1 = allocate_matrix_as_1d_array(number_of_rows, number_of_columns);
        double *matrix_2 = allocate_matrix_as_1d_array(number_of_rows, number_of_columns);
        double *matrix_3 = allocate_matrix_as_1d_array(number_of_rows, number_of_columns);

        populate_matrix_as_1d_array_with_random_values(matrix_1, number_of_rows, number_of_columns);
        populate_matrix_as_1d_array_with_random_values(matrix_2, number_of_rows, number_of_columns);
        populate_matrix_as_1d_array_with_zeros(matrix_3, number_of_rows, number_of_columns);

        double start_time_in_seconds = get_current_number_of_seconds();
        matrix_multiplication_sequential_03(matrix_1, matrix_2, matrix_3, number_of_rows, number_of_columns);
        double end_time_in_seconds = get_current_number_of_seconds();

        double run_duration_in_seconds = end_time_in_seconds - start_time_in_seconds;

        unsigned long long total_number_of_multiplications = (unsigned long long)number_of_rows * number_of_rows * number_of_rows;
        unsigned long long total_number_of_additions = (unsigned long long)number_of_rows * number_of_rows * (number_of_rows - 1);
        unsigned long long total_number_of_floating_point_operations = total_number_of_multiplications + total_number_of_additions;

        double total_number_of_floating_point_operations_per_second = (double)total_number_of_floating_point_operations / run_duration_in_seconds;
        double total_number_of_billions_of_floating_point_operations_per_second = total_number_of_floating_point_operations_per_second / ONE_BILLION_FLOATING_POINT_OPERATIONS;

        fprintf(benchmarking_result_file, "%d,%d,%d,%.5f,%.5f\n", current_run_index + 1, number_of_rows, number_of_columns, run_duration_in_seconds, total_number_of_billions_of_floating_point_operations_per_second);

        free(matrix_1);
        free(matrix_2);
        free(matrix_3);

        printf("Run %d complete.\n", current_run_index + 1);
    }
    printf("Benchmarking of sequential implementation number 03 ended.\n");
    
    fclose(benchmarking_result_file);
}

void benchmark_parallel_03(int number_of_runs, int number_of_rows, int number_of_columns)
{
    char benchmarking_result_file_name[128];
    snprintf(benchmarking_result_file_name, sizeof(benchmarking_result_file_name), "benchmarking_results/parallel/parallel_03/parallel_03_%d_%d.csv", number_of_rows, number_of_columns);

    FILE *benchmarking_result_file = fopen(benchmarking_result_file_name, "w");
    if (!benchmarking_result_file)
    {
        fprintf(stderr, "Bencmarking result file opening failed.\n");
        exit(1);
    }

    printf("Benchmarking of parallel implementation number 03 started.\n");
    fprintf(benchmarking_result_file, "run_index,number_of_matrix_rows,number_of_matrix_columns,run_duration_in_seconds,gflops\n");

    for (int current_run_index = 0; current_run_index < number_of_runs; current_run_index++)
    {
        double *matrix_1 = allocate_matrix_as_1d_array(number_of_rows, number_of_columns);
        double *matrix_2 = allocate_matrix_as_1d_array(number_of_rows, number_of_columns);
        double *matrix_3 = allocate_matrix_as_1d_array(number_of_rows, number_of_columns);

        populate_matrix_as_1d_array_with_random_values(matrix_1, number_of_rows, number_of_columns);
        populate_matrix_as_1d_array_with_random_values(matrix_2, number_of_rows, number_of_columns);
        populate_matrix_as_1d_array_with_zeros(matrix_3, number_of_rows, number_of_columns);

        double start_time_in_seconds = get_current_number_of_seconds();
        matrix_multiplication_parallel_03(matrix_1, matrix_2, matrix_3, number_of_rows, number_of_columns);
        double end_time_in_seconds = get_current_number_of_seconds();

        double run_duration_in_seconds = end_time_in_seconds - start_time_in_seconds;

        unsigned long long total_number_of_multiplications = (unsigned long long)number_of_rows * number_of_rows * number_of_rows;
        unsigned long long total_number_of_additions = (unsigned long long)number_of_rows * number_of_rows * (number_of_rows - 1);
        unsigned long long total_number_of_floating_point_operations = total_number_of_multiplications + total_number_of_additions;

        double total_number_of_floating_point_operations_per_second = (double)total_number_of_floating_point_operations / run_duration_in_seconds;
        double total_number_of_billions_of_floating_point_operations_per_second = total_number_of_floating_point_operations_per_second / ONE_BILLION_FLOATING_POINT_OPERATIONS;

        fprintf(benchmarking_result_file, "%d,%d,%d,%.5f,%.5f\n", current_run_index + 1, number_of_rows, number_of_columns, run_duration_in_seconds, total_number_of_billions_of_floating_point_operations_per_second);

        free(matrix_1);
        free(matrix_2);
        free(matrix_3);

        printf("Run %d complete.\n", current_run_index + 1);
    }
    printf("Benchmarking of parallel implementation number 03 ended.\n");
    
    fclose(benchmarking_result_file);
}

int main(int number_of_arguments, char **arguments)
{
    if (number_of_arguments < 4)
    {
        exit(1);
    }

    srand(42);

    int number_of_runs = atoi(arguments[1]);
    int number_of_rows = atoi(arguments[2]);
    int number_of_columns = atoi(arguments[3]);

    benchmark_sequential_01(number_of_runs, number_of_rows, number_of_columns);
    benchmark_sequential_02(number_of_runs, number_of_rows, number_of_columns);
    benchmark_sequential_03(number_of_runs, number_of_rows, number_of_columns);

    omp_set_num_threads(TOTAL_NUMBER_OF_PHYSICAL_PROCESSOR_CORES);

    benchmark_parallel_03(number_of_runs, number_of_rows, number_of_columns);

    return 0;
}