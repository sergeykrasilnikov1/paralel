#include <stdio.h>
#include <omp.h>
#include <time.h>
#include <inttypes.h>
#include <stdlib.h>


double wtime() {
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return (double)time.tv_sec + (double)time.tv_nsec * 1e-9;
}


void matrix_vector_product_omp(double *a, double *b, double *c, int m, int n, int n_threads) {
#pragma omp parallel for num_threads(n_threads)
    for (int i = 0; i < m; i++) {
        c[i] = 0.0;
        for (int j = 0; j < n; j++)
            c[i] += a[i * n + j] * b[j];
    }
}


double run_parallel(int m, int n,int n_threads) {
    double *a, *b, *c;

    // Allocate memory for 2-d array a[m, n]
    a = (double*)malloc(sizeof(*a) * m * n);
    b = (double*)malloc(sizeof(*b) * n);
    c = (double*)malloc(sizeof(*c) * m);
    #pragma omp parallel for num_threads(40)
    for (int i = 0; i < m; i++) {
        b[j] = j;
        for (int j = 0; j < n; j++)
            a[i * n + j] = i + j;
    }

        

    double t = wtime();
    matrix_vector_product_omp(a, b, c, m, n, n_threads);
    t = wtime() - t;

    printf("Elapsed time (parallel): %.6f sec.\n", t);

    free(a);
    free(b);
    free(c);
    return t;
}

void matrix_vector_product(double *a, double *b, double *c, int m, int n)
{
    for (int i = 0; i < m; i++) {
        c[i] = 0.0;
        for (int j = 0; j < n; j++)
            c[i] += a[i * n + j] * b[j];

    }
}

double run_serial(int m, int n) {
    double *a, *b, *c;
    a = (double*)malloc(sizeof(*a) * m * n);
    b = (double*)malloc(sizeof(*b) * n);
    c = (double*)malloc(sizeof(*c) * m);
    for (int i = 0; i < m; i++) {
        b[j] = j;
        for (int j = 0; j < n; j++)
            a[i * n + j] = i + j;

    }
        
    double t = wtime();
    matrix_vector_product(a, b, c, m, n);
    t = wtime() - t;
    printf("Elapsed time (serial): %.6f sec.\n", t);
    free(a);
    free(b);
    free(c);
    return t;
}

void tests(int m, int n) {
    double serial_time = run_serial(m, n);
    int threads[] = { 2, 4, 7, 8, 16, 20, 40 };
    for (int i = 0; i < 7; i++) {
        printf("test with %d threads\n", threads[i]);
        double parallel_time = run_parallel(m, n, threads[i]); // test parallel product
//        printf("Elapsed time(serial realization)  %lf sec.\n", parallel_time);
        printf("Speedup  %lf\n", serial_time/parallel_time);
    }
}



int main(int argc, char **argv) {
    int m = 20000, n = m;
    printf("Matrix-vector product (c[m] = a[m, n] * b[n]; m = %d, n = %d)\n", m, n);
    printf("Memory used: %" PRIu64 " MiB\n", ((m * n + m + n) * sizeof(double))  >> 20);
    tests(m, n);
    m = 40000, n=m;
    printf("Matrix-vector product (c[m] = a[m, n] * b[n]; m = %d, n = %d)\n", m, n);
    tests(m,n);
    return 0;
}
