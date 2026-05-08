#define _POSIX_C_SOURCE 199309L

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#define NUMBER_OF_NANOSECONDS_IN_ONE_SECOND 1e9
#define ONE_BILLION_FLOATING_POINT_OPERATIONS 1e9
#define TOTAL_NUMBER_OF_PHYSICAL_PROCESSOR_CORES 4

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

// The number of threads per block dimension.
// A 16x16 block = 256 threads, which is a sweet spot:
// enough parallelism, fits in shared memory, divides most matrix sizes.
#define TILE_SIZE 16

static double get_current_number_of_seconds()
{
    struct timespec time_specification;
    clock_gettime(CLOCK_MONOTONIC, &time_specification);
    return time_specification.tv_sec + time_specification.tv_nsec / NUMBER_OF_NANOSECONDS_IN_ONE_SECOND;
}

static void* safe_malloc(size_t size_in_bytes)
{
    void *pointer = malloc(size_in_bytes);
    if (!pointer) { fprintf(stderr, "Memory allocation failed.\n"); exit(1); }
    return pointer;
}

static double *allocate_matrix_as_1d_array(int number_of_rows, int number_of_columns)
{
    size_t total = (size_t)number_of_rows * number_of_columns * sizeof(double);
    return (double *)safe_malloc(total);
}

static void populate_matrix_as_1d_array_with_random_values(double *matrix, int number_of_rows, int number_of_columns)
{
    unsigned long total = (unsigned long)number_of_rows * number_of_columns;
    for (unsigned long i = 0; i < total; i++)
        matrix[i] = (double)rand() / RAND_MAX;
}

static void populate_matrix_as_1d_array_with_zeros(double *matrix, int number_of_rows, int number_of_columns)
{
    memset(matrix, 0, (size_t)number_of_rows * number_of_columns * sizeof(double));
}

// ─── GPU memory helpers ───────────────────────────────────────────────────────
// GPU (device) memory is completely separate from CPU (host) memory.
// You can't just pass a CPU pointer to a kernel — you must:
//   1. cudaMalloc  — allocate on GPU
//   2. cudaMemcpy  — copy data CPU→GPU
//   3. launch kernel
//   4. cudaMemcpy  — copy result GPU→CPU
//   5. cudaFree    — free GPU memory

static double *allocate_matrix_on_gpu(int number_of_rows, int number_of_columns)
{
    double *device_pointer;
    size_t size = (size_t)number_of_rows * number_of_columns * sizeof(double);
    CUDA_CHECK(cudaMalloc((void **)&device_pointer, size));
    return device_pointer;
}

static void copy_matrix_cpu_to_gpu(double *device_dst, double *host_src, int number_of_rows, int number_of_columns)
{
    size_t size = (size_t)number_of_rows * number_of_columns * sizeof(double);
    CUDA_CHECK(cudaMemcpy(device_dst, host_src, size, cudaMemcpyHostToDevice));
}

static void copy_matrix_gpu_to_cpu(double *host_dst, double *device_src, int number_of_rows, int number_of_columns)
{
    size_t size = (size_t)number_of_rows * number_of_columns * sizeof(double);
    CUDA_CHECK(cudaMemcpy(host_dst, device_src, size, cudaMemcpyDeviceToHost));
}

// ─── THE CUDA KERNEL ──────────────────────────────────────────────────────────
// __global__ means: this function runs on the GPU, called from the CPU.
// Every thread executes this function simultaneously (SIMT model).
//
// Thread identification:
//   blockIdx  — which block this thread is in  (in the grid)
//   threadIdx — which thread within the block
//   blockDim  — how many threads per block dimension
//
// So the global row/col this thread is responsible for:
//   row = blockIdx.y * blockDim.y + threadIdx.y
//   col = blockIdx.x * blockDim.x + threadIdx.x

__global__ void matmul_tiled_kernel(
    const double * __restrict__ A,   // input  matrix, row-major, N×N
    const double * __restrict__ B,   // input  matrix, row-major, N×N
    double       * __restrict__ C,   // output matrix, row-major, N×N
    int N)                           // matrix dimension
{
    // __shared__ memory lives in a fast, per-block scratchpad (~48KB).
    // All threads in the block share it. Latency ~5 cycles vs ~400 for global.
    __shared__ double tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ double tile_B[TILE_SIZE][TILE_SIZE];

    // This thread's position in the output matrix
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    // The accumulator.
    // This lives in a register — the fastest possible memory, private to each thread. 
    // Over the course of the kernel, each thread accumulates its dot product result here, one tile at a time. 
    // At the end, this holds the final value for C[row][col].
    double partial_sum = 0.0;

    // Iterate over tiles along the shared K-dimension.
    // Instead of reading A[row, 0..N] and B[0..N, col] all from global memory,
    // we cooperatively load TILE_SIZE columns of A and TILE_SIZE rows of B
    // into shared memory, compute the partial dot product, then slide the tile.
    int number_of_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (int tile_index = 0; tile_index < number_of_tiles; tile_index++)
    {
        // ── Load phase ────────────────────────────────────────────────────────
        // Each thread loads one element of tile_A and one element of tile_B.
        // Threads with threadIdx.y=0..15, threadIdx.x=0..15 collectively fill
        // a 16×16 tile in a single step.
        //
        // Bounds check: if the matrix size isn't a multiple of TILE_SIZE,
        // some threads are "out of bounds" — they load 0.0 instead (padding).

        int a_col = tile_index * TILE_SIZE + threadIdx.x;
        int b_row = tile_index * TILE_SIZE + threadIdx.y;

        tile_A[threadIdx.y][threadIdx.x] = (row < N && a_col < N)
                                            ? A[row * N + a_col]
                                            : 0.0;

        tile_B[threadIdx.y][threadIdx.x] = (b_row < N && col < N)
                                            ? B[b_row * N + col]
                                            : 0.0;

        // ── Synchronization barrier ───────────────────────────────────────────
        // All threads in the block must finish loading before any thread starts
        // computing. Without this, a fast thread might read a tile element that
        // a slow thread hasn't written yet.
        __syncthreads();

        // ── Compute phase ─────────────────────────────────────────────────────
        // Now accumulate the dot product from shared memory (fast).
        for (int k = 0; k < TILE_SIZE; k++)
        {
            partial_sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
        }

        // ── Second barrier ────────────────────────────────────────────────────
        // All threads must finish computing before any thread overwrites the
        // shared tile in the next loop iteration.
        __syncthreads();
    }

    // Write result — only if this thread actually maps to a valid output cell.
    if (row < N && col < N)
    {
        C[row * N + col] = partial_sum;
    }
}

// ─── Host-side wrapper ────────────────────────────────────────────────────────
// This is called from CPU code. It handles memory transfer and kernel launch.
void matrix_multiplication_cuda(double *host_A, double *host_B, double *host_C,
                                 int number_of_rows, int number_of_columns)
{
    int N = number_of_rows; // we assume square matrices

    // 1. Allocate GPU memory for all three matrices
    double *device_A = allocate_matrix_on_gpu(N, N);
    double *device_B = allocate_matrix_on_gpu(N, N);
    double *device_C = allocate_matrix_on_gpu(N, N);

    // 2. Copy input matrices from CPU to GPU
    copy_matrix_cpu_to_gpu(device_A, host_A, N, N);
    copy_matrix_cpu_to_gpu(device_B, host_B, N, N);

    // 3. Configure the kernel launch
    //
    // dim3 is a 3-component struct (x, y, z). For 2D problems, we use x and y.
    //
    // block_dim: 16×16 = 256 threads per block.
    //   This is a standard choice. Max is 1024. Powers of 2 work best.
    //
    // grid_dim: how many blocks we need to cover the full N×N matrix.
    //   ceiling division: (N + TILE_SIZE - 1) / TILE_SIZE
    //   e.g. N=1000, TILE_SIZE=16 → 63 blocks per dimension → 63×63 = 3969 blocks
    //   Each block covers a 16×16 patch of C.

    dim3 block_dim(TILE_SIZE, TILE_SIZE);
    dim3 grid_dim(
        (N + TILE_SIZE - 1) / TILE_SIZE,   // blocks in X (covers columns)
        (N + TILE_SIZE - 1) / TILE_SIZE    // blocks in Y (covers rows)
    );

    // 4. Launch the kernel
    //    Syntax: kernel<<<grid_dim, block_dim>>>(args...)
    //    This is asynchronous — CPU continues immediately after dispatch.
    matmul_tiled_kernel<<<grid_dim, block_dim>>>(device_A, device_B, device_C, N);

    // 5. Check for kernel launch errors (wrong args, OOM on GPU, etc.)
    CUDA_CHECK(cudaGetLastError());

    // 6. Wait for GPU to finish (cudaMemcpy below is blocking, but being
    //    explicit here makes timing more accurate)
    CUDA_CHECK(cudaDeviceSynchronize());

    // 7. Copy result back from GPU to CPU
    copy_matrix_gpu_to_cpu(host_C, device_C, N, N);

    // 8. Free GPU memory
    CUDA_CHECK(cudaFree(device_A));
    CUDA_CHECK(cudaFree(device_B));
    CUDA_CHECK(cudaFree(device_C));
}

// ─── Benchmark ───────────────────────────────────────────────────────────────
void benchmark_cuda(int number_of_runs, int number_of_rows, int number_of_columns)
{
    char benchmarking_result_file_name[128];
    snprintf(benchmarking_result_file_name, sizeof(benchmarking_result_file_name), "benchmarking_results/parallel/cuda/cuda_%d_%d.csv",
             number_of_rows, number_of_columns);

    FILE *benchmarking_result_file = fopen(benchmarking_result_file_name, "w");
    if (!benchmarking_result_file)
    {
        fprintf(stderr, "Benchmarking result file opening failed.\n");
        exit(1);
    }

    printf("Benchmarking of CUDA implementation started.\n");
    fprintf(benchmarking_result_file,
            "run_index,number_of_matrix_rows,number_of_matrix_columns,run_duration_in_seconds,gflops\n");

    // Warm-up run: the first CUDA call on a process pays a one-time driver
    // initialization cost (~100ms). Run once and discard so it doesn't
    // contaminate your benchmark numbers.
    {
        double *warmup_A = allocate_matrix_as_1d_array(number_of_rows, number_of_columns);
        double *warmup_B = allocate_matrix_as_1d_array(number_of_rows, number_of_columns);
        double *warmup_C = allocate_matrix_as_1d_array(number_of_rows, number_of_columns);
        populate_matrix_as_1d_array_with_random_values(warmup_A, number_of_rows, number_of_columns);
        populate_matrix_as_1d_array_with_random_values(warmup_B, number_of_rows, number_of_columns);
        populate_matrix_as_1d_array_with_zeros(warmup_C, number_of_rows, number_of_columns);
        matrix_multiplication_cuda(warmup_A, warmup_B, warmup_C, number_of_rows, number_of_columns);
        free(warmup_A); free(warmup_B); free(warmup_C);
        printf("CUDA warm-up run complete.\n");
    }

    for (int current_run_index = 0; current_run_index < number_of_runs; current_run_index++)
    {
        double *matrix_1 = allocate_matrix_as_1d_array(number_of_rows, number_of_columns);
        double *matrix_2 = allocate_matrix_as_1d_array(number_of_rows, number_of_columns);
        double *matrix_3 = allocate_matrix_as_1d_array(number_of_rows, number_of_columns);

        populate_matrix_as_1d_array_with_random_values(matrix_1, number_of_rows, number_of_columns);
        populate_matrix_as_1d_array_with_random_values(matrix_2, number_of_rows, number_of_columns);
        populate_matrix_as_1d_array_with_zeros(matrix_3, number_of_rows, number_of_columns);

        // NOTE: we include the H→D and D→H transfers in the timing.
        // This measures total wall-clock cost, which is the fair comparison
        // against CPU versions (they also include memory access).
        double start_time_in_seconds = get_current_number_of_seconds();
        matrix_multiplication_cuda(matrix_1, matrix_2, matrix_3, number_of_rows, number_of_columns);
        double end_time_in_seconds = get_current_number_of_seconds();

        double run_duration_in_seconds = end_time_in_seconds - start_time_in_seconds;

        unsigned long long total_number_of_multiplications = (unsigned long long)number_of_rows * number_of_rows * number_of_rows;
        unsigned long long total_number_of_additions = (unsigned long long)number_of_rows * number_of_rows * (number_of_rows - 1);
        unsigned long long total_number_of_floating_point_operations = total_number_of_multiplications + total_number_of_additions;

        double gflops = ((double)total_number_of_floating_point_operations / run_duration_in_seconds) / ONE_BILLION_FLOATING_POINT_OPERATIONS;

        fprintf(benchmarking_result_file, "%d,%d,%d,%.5f,%.5f\n", current_run_index + 1, number_of_rows, number_of_columns, run_duration_in_seconds, gflops);

        free(matrix_1);
        free(matrix_2);
        free(matrix_3);

        printf("Run %d complete.\n", current_run_index + 1);
    }

    printf("Benchmarking of CUDA implementation ended.\n");
    fclose(benchmarking_result_file);
}

int main(int number_of_arguments, char **arguments)
{
    if (number_of_arguments < 4) exit(1);

    srand(42);

    int number_of_runs    = atoi(arguments[1]);
    int number_of_rows    = atoi(arguments[2]);
    int number_of_columns = atoi(arguments[3]);

    benchmark_cuda(number_of_runs, number_of_rows, number_of_columns);

    return 0;
}