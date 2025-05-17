#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <unistd.h>

#define SIZE 1000
#define LOG_FILE "output.log"

double **A, **B, **BT, **C;
int enable_io = 1; // 預設啟用 I/O

void allocate_matrices()
{
    A = (double **)malloc(SIZE * sizeof(double *));
    B = (double **)malloc(SIZE * sizeof(double *));
    BT = (double **)malloc(SIZE * sizeof(double *));
    C = (double **)malloc(SIZE * sizeof(double *));
    for (int i = 0; i < SIZE; i++)
    {
        A[i] = (double *)malloc(SIZE * sizeof(double));
        B[i] = (double *)malloc(SIZE * sizeof(double));
        BT[i] = (double *)malloc(SIZE * sizeof(double));
        C[i] = (double *)malloc(SIZE * sizeof(double));
    }
}

void init_matrices()
{
    for (int i = 0; i < SIZE; i++)
        for (int j = 0; j < SIZE; j++)
        {
            A[i][j] = rand() % 100;
            B[i][j] = rand() % 100;
            C[i][j] = 0.0;
        }
}

void memory_bound_sequential()
{
    FILE *fp = enable_io ? fopen("seq_" LOG_FILE, "w") : NULL;
    if (enable_io && !fp)
    {
        perror("Cannot open file");
        return;
    }

    for (int i = 0; i < SIZE; i++)
        for (int j = 0; j < SIZE; j++)
        {
            for (int k = 0; k < SIZE; k++)
            {
                C[i][j] += A[i][k] * B[k][j];
            }
            if (enable_io && j % 100 == 0)
            {
                usleep(5000);
                fprintf(fp, "Sequential processed row %d column %d\n", i, j);
            }
        }

    if (enable_io)
        fclose(fp);
}

void memory_bound_with_openmp()
{
    FILE *fp = enable_io ? fopen(LOG_FILE, "w") : NULL;
    if (enable_io && !fp)
    {
        perror("Cannot open file");
        return;
    }

#pragma omp parallel
    {
#pragma omp for collapse(2) schedule(dynamic)
        for (int i = 0; i < SIZE; i++)
            for (int j = 0; j < SIZE; j++)
            {
                for (int k = 0; k < SIZE; k++)
                    C[i][j] += A[i][k] * B[k][j];

                if (enable_io && j % 100 == 0)
                {
                    usleep(5000);
                    fprintf(fp, "Thread %d processed row %d column %d\n",
                            omp_get_thread_num(), i, j);
                }
            }
    }

    if (enable_io)
        fclose(fp);
}

void reset_C()
{
    for (int i = 0; i < SIZE; i++)
        for (int j = 0; j < SIZE; j++)
            C[i][j] = 0.0;
}

int main(int argc, char *argv[])
{
    if (argc > 1)
    {
        if (strcmp(argv[1], "--no-io") == 0)
        {
            enable_io = 0;
            printf("I/O disabled.\n");
        }
        else if (strcmp(argv[1], "--io") == 0)
        {
            enable_io = 1;
            printf("I/O enabled.\n");
        }
        else
        {
            printf("Unknown option: %s\n", argv[1]);
            printf("Usage: %s [--io | --no-io]\n", argv[0]);
            return 1;
        }
    }

    int max_threads = omp_get_max_threads();
    printf("Running with %d OpenMP threads...\n", max_threads);

    allocate_matrices();
    init_matrices();

    double start_seq = omp_get_wtime();
    memory_bound_sequential();
    double end_seq = omp_get_wtime();
    double time_seq = end_seq - start_seq;
    printf("Sequential Time: %.3f seconds\n", time_seq);

    reset_C();

    double start_omp = omp_get_wtime();
    memory_bound_with_openmp();
    double end_omp = omp_get_wtime();
    double time_omp = end_omp - start_omp;
    printf("OpenMP Time with %d threads: %.3f seconds\n", max_threads, time_omp);

    double speedup = time_seq / time_omp;
    double improvement = (1.0 - time_omp / time_seq) * 100.0;
    printf("Speedup: %.2fx\n", speedup);
    printf("Improvement: %.2f%%\n", improvement);

    return 0;
}
