#define _POSIX_C_SOURCE 199309L

//#include "Maxfiles.h"
//#include <MaxSLiCInterface.h>

/**
 * MatMulCpuCode.c
 *
 * CPU host code for the Maxeler DFE matrix multiplication accelerator.
 *
 * Steps:
 *   1. Allocate and initialise matrices A, B (NxN, float32).
 *   2. Compute the CPU reference result for verification.
 *   3. Load matrix B into DFE FMem via mapped memory write.
 *   4. Build the DFE input stream for A:
 *        Each A[i][k] is repeated N times (once per output column j).
 *        Total stream length = N*N*N elements.
 *   5. Run DFE action: MatMul(ticks, a_stream, c_stream).
 *   6. Verify DFE output against CPU reference.
 *   7. Print timing and PASS/FAIL.
 *
 * Compile via the Makefile in RunRules/Simulation/ or RunRules/DFE/.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* Auto-generated SLiC header from MaxCompiler */
//#include "MatMul.h"

/* ------------------------------------------------------------------ */
/* Configuration — must match MatMulManager.maxj                       */
/* ------------------------------------------------------------------ */
#define N        64
#define EPSILON  1e-3f

/* ------------------------------------------------------------------ */
/* Fill a matrix with pseudo-random floats in [0, scale)              */
/* ------------------------------------------------------------------ */
static void fill_matrix(float *M, int rows, int cols, float scale) {
    for (int i = 0; i < rows * cols; i++) {
        M[i] = scale * ((float)rand() / (float)RAND_MAX);
    }
}

/* ------------------------------------------------------------------ */
/* CPU reference: naive O(N^3) matrix multiply for correctness check  */
/* ------------------------------------------------------------------ */
static void cpu_matmul(const float *A, const float *B, float *C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

/* ------------------------------------------------------------------ */
/* Verify DFE result vs CPU reference                                  */
/* ------------------------------------------------------------------ */
static int verify(const float *ref, const float *dfe, int size) {
    int errors = 0;
    float max_err = 0.0f;
    int max_err_idx = -1;

    for (int i = 0; i < size; i++) {
        float diff = fabsf(ref[i] - dfe[i]);
        if (diff > EPSILON) {
            if (errors < 5) {
                fprintf(stderr,
                    "  Mismatch at [%d]: CPU=%.6f  DFE=%.6f  diff=%.3e\n",
                    i, ref[i], dfe[i], diff);
            }
            errors++;
        }
        if (diff > max_err) { max_err = diff; max_err_idx = i; }
    }

    if (errors == 0) {
        printf("  Verification PASSED. Max absolute error = %.2e\n", max_err);
    } else {
        printf("  Verification FAILED: %d/%d elements exceeded tolerance %.1e\n",
               errors, size, (double)EPSILON);
        printf("  Largest error = %.3e at flat index %d\n", max_err, max_err_idx);
    }
    return errors;
}

/* ------------------------------------------------------------------ */
/* Build the DFE input stream from matrix A.                          */
/*                                                                      */
/* The kernel inner loop is k=0..N-1, outer loop j=0..N-1.            */
/* For a fixed row i and inner index k, A[i][k] is consumed N times   */
/* (once per output column j). The host must therefore repeat each     */
/* A[i][k] N consecutive times before advancing k.                    */
/*                                                                      */
/* Stream layout:                                                        */
/*   A[0][0] x N,  A[0][1] x N,  ..., A[0][N-1] x N,                 */
/*   A[1][0] x N,  A[1][1] x N,  ..., A[1][N-1] x N,                 */
/*   ...                                                                */
/*   A[N-1][0] x N, ...,          A[N-1][N-1] x N                     */
/*                                                                      */
/* Total length = N * N * N elements (= total kernel ticks).           */
/* ------------------------------------------------------------------ */
static void build_a_stream(const float *A, float *stream, int n) {
    int idx = 0;
    for (int i = 0; i < n; i++) {        /* Row of A          */
        for (int k = 0; k < n; k++) {    /* Column of A       */
            float val = A[i * n + k];
            for (int j = 0; j < n; j++) { /* Repeat N times   */
                stream[idx++] = val;
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* Wall-clock timer                                                     */
/* ------------------------------------------------------------------ */
static double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* ------------------------------------------------------------------ */
/* main                                                                 */
/* ------------------------------------------------------------------ */
int main(void) {
    printf("=== Maxeler DFE Matrix Multiplication  (N = %d) ===\n\n", N);

    const int  mat_size  = N * N;
    const long ticks     = (long)N * N * N;

    /* Allocate all buffers */
    float *A        = (float *)malloc(mat_size * sizeof(float));
    float *B        = (float *)malloc(mat_size * sizeof(float));
    float *C_cpu    = (float *)malloc(mat_size * sizeof(float));
    float *C_dfe    = (float *)malloc(mat_size * sizeof(float));
    float *a_stream = (float *)malloc(ticks    * sizeof(float));

    if (!A || !B || !C_cpu || !C_dfe || !a_stream) {
        fprintf(stderr, "ERROR: memory allocation failed\n");
        return EXIT_FAILURE;
    }

    /* Initialise A and B with reproducible values */
    srand(42);
    fill_matrix(A, N, N, 1.0f);
    fill_matrix(B, N, N, 1.0f);
    memset(C_cpu, 0, mat_size * sizeof(float));
    memset(C_dfe, 0, mat_size * sizeof(float));

    /* -------------------------------------------------------------- */
    /* Step 1: CPU reference                                           */
    /* -------------------------------------------------------------- */
    printf("[1] CPU reference computation...\n");
    double t0 = get_time_sec();
    cpu_matmul(A, B, C_cpu, N);
    double cpu_time = get_time_sec() - t0;
    printf("    Time = %.4f s\n\n", cpu_time);

    /* -------------------------------------------------------------- */
    /* Step 2: Build DFE input stream                                  */
    /* -------------------------------------------------------------- */
    printf("[2] Building DFE input stream (%ld ticks, %.2f MB)...\n",
           ticks, (double)(ticks * sizeof(float)) / (1024.0 * 1024.0));
    build_a_stream(A, a_stream, N);
    printf("    Done.\n\n");

    /* -------------------------------------------------------------- */
    /* Step 3: Load the DFE (or connect to simulation)                 */
    /* -------------------------------------------------------------- */
    printf("[3] Loading DFE...\n");
    max_file_t   *max_file   = MatMul_init();
    max_engine_t *max_engine = max_load(max_file, "*");
    printf("    DFE ready.\n\n");

    /* -------------------------------------------------------------- */
    /* Step 4: Write matrix B into DFE on-chip FMem                   */
    /*                                                                  */
    /* The kernel declared:  memB.mapToCPU("memB")                     */
    /* MaxCompiler generates: MatMul_writeLMem_memB(engine, offset,    */
    /*                            size_in_bytes, data_ptr)             */
    /* We write all N*N floats starting at byte offset 0.              */
    /* -------------------------------------------------------------- */
    printf("[4] Loading matrix B into DFE FMem (%d floats = %.2f KB)...\n",
           mat_size, (double)(mat_size * sizeof(float)) / 1024.0);
    MatMul_writeLMem_memB(max_engine, 0, mat_size * sizeof(float), B);
    printf("    Done.\n\n");

    /* -------------------------------------------------------------- */
    /* Step 5: Run DFE action                                          */
    /*                                                                  */
    /* SLiC auto-generated signature:                                  */
    /*   void MatMul(                                                   */
    /*       uint64_t  ticks_MatMulKernel,                             */
    /*       const void *instream_a,                                   */
    /*       void       *outstream_c                                   */
    /*   );                                                             */
    /*                                                                  */
    /* outstream_c receives N*N float32 values (controlled output).    */
    /* The SLiC runtime handles the gaps from the enable signal.       */
    /* -------------------------------------------------------------- */
    printf("[5] Running DFE action (%ld ticks)...\n", ticks);
    double t1 = get_time_sec();

    MatMul(
        (uint64_t)ticks,  /* ticks_MatMulKernel */
        a_stream,         /* instream_a         */
        C_dfe             /* outstream_c        */
    );

    double dfe_time = get_time_sec() - t1;
    printf("    DFE time = %.4f s\n\n", dfe_time);

    /* -------------------------------------------------------------- */
    /* Step 6: Verify                                                   */
    /* -------------------------------------------------------------- */
    printf("[6] Verifying result...\n");
    int errors = verify(C_cpu, C_dfe, mat_size);

    /* -------------------------------------------------------------- */
    /* Step 7: Clean up                                                 */
    /* -------------------------------------------------------------- */
    max_unload(max_engine);
    max_file_free(max_file);
    free(A); free(B); free(C_cpu); free(C_dfe); free(a_stream);

    /* -------------------------------------------------------------- */
    /* Performance summary                                              */
    /* -------------------------------------------------------------- */
    double flops     = 2.0 * (double)N * (double)N * (double)N;
    double gf_cpu    = flops / (cpu_time * 1e9);
    double gf_dfe    = flops / (dfe_time * 1e9);

    printf("\n=== Summary ===\n");
    printf("  Matrix size : %d x %d\n",   N, N);
    printf("  Total ticks : %ld\n",       ticks);
    printf("  CPU time    : %.4f s  (%.4f GFLOP/s)\n", cpu_time, gf_cpu);
    printf("  DFE time    : %.4f s  (%.4f GFLOP/s)\n", dfe_time, gf_dfe);
    printf("  Speedup     : %.2fx\n",     cpu_time / dfe_time);
    printf("  Result      : %s\n",        errors == 0 ? "PASS ✓" : "FAIL ✗");

    return (errors == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}