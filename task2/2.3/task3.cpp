#include <stdio.h>
#include <omp.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>

double epsilon = 0.00001;
double tau = 0.01;
int threads_number = 0;
double cpuSecond()
{
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ((double)ts.tv_sec + (double)ts.tv_nsec * 1.e-9);
}

void parallel_product(double* A, double* b, double* c, int n, int m, int n_threads) {
        #pragma omp parallel  for num_threads(n_threads)
        for (int i = 0; i < n; i++) {
            c[i] = 0;
            for (int j = 0; j < m; j++)
                c[i] += A[i * n + j] * b[j];
        }
}

void serial_product(double* A, double* b, double* c, int n, int m) {
    for (int i = 0; i < n; i++) {
        c[i] = 0;
        for (int j = 0; j < m; j++)
            c[i] += A[i * n + j] * b[j];
    }
}

double vector_length(double* vec, int n) {
    double sum = 0;
    for (int i = 0; i < n; i++)
        sum += vec[i] * vec[i];
    return sqrt(sum);
}

void serial_method(double* A, double* x, double* b, int n) {
    double* x_new = (double*)malloc(sizeof(double) * n);
    double end_coef = 1;
    double t = cpuSecond();
    while (end_coef > epsilon) {
        serial_product(A, x, x_new, n, n);
        double x_offset_length = 0;
        for (int i = 0; i < n; i++) {
            x[i] = x[i] - tau * (x_new[i] - b[i]);
            x_offset_length += (x_new[i] - b[i]) * (x_new[i] - b[i]);
        }
        end_coef = sqrt(x_offset_length) / vector_length(b, n);;
    }
    t = cpuSecond() - t;
    printf("%.12f\n", t);
    free(x_new);
}

void parallel_method1(double* A, double* x, double* b, int n, int n_threads) {
    double* x_new = (double*)malloc(sizeof(double) * n);
    double end_coef = 1;
    double t = cpuSecond();
    while (end_coef > epsilon) {
        parallel_product(A, x, x_new, n, n, n_threads);
        double different_length = 0;
        for (int i = 0; i < n; i++) {
            x[i] = x[i] - tau * (x_new[i] - b[i]);
            different_length += (x_new[i] - b[i]) * (x_new[i] - b[i]);
        }
        end_coef = sqrt(different_length) / vector_length(b, n);;
    }
    t = cpuSecond() - t;
    printf("%.12f\n", t);
    free(x_new);
}

void parallel_method2(double* A, double* x, double* b, int n, int n_threads) {

    double t = cpuSecond();
    double* x_new = (double*)malloc(sizeof(double) * n);
    double end_coef=1;
    double different_length=0;
    #pragma omp parallel num_threads(n_threads)
    {


        while (end_coef > epsilon) {


    #pragma omp for schedule(guided, 20) reduction(+:different_length)
            for (int i = 0; i < n; i++) {
                x_new[i] = 0;
                for (int j = 0; j < n; j++)
                    x_new[i] += A[i * n + j] * x[j];
                different_length += (x_new[i] - b[i]) * (x_new[i] - b[i]);

            }
    #pragma omp for
            for (int i = 0; i < n; i++) {
                x[i] = x[i] - tau * (x_new[i] - b[i]);
            }

    #pragma omp single
            {
                end_coef = sqrt(different_length) / vector_length(b, n);
            }

        }
    }
    t = cpuSecond() - t;
    printf("%.12f\n", t);
    free(x_new);
}




int main() {
    int n = 4000;
    double* A = (double*)malloc(sizeof(double) * n * n);
    double* b = (double*)malloc(sizeof(double) * n);
    double* x = (double*)malloc(sizeof(double) * n);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n;j ++) {
            if (i == j) A[i * n + j] = 2.0;
            else A[i * n + j] = 1.0;
        }
        x[i] = 0;
    }
    for (int i = 0; i < n; i++)
        b[i] = n + 1;


     printf("serial time:\n");
    serial_method(A, x, b, n);
     for (int i = 0; i < n; i++) {
         x[i] = 0;

     }
    int threads[] = {  2, 4, 8, 16, 20, 40 };
    for (int j = 0; j < 6; j++) {
        threads_number = threads[j];
        printf("thread_number = %d\n", threads_number);
        printf("first method parallel time:\n");
        parallel_method1(A, x, b, n, threads_number);
        for (int i = 0; i < n; i++) {
            x[i] = 0;
        }
        printf("second method parallel time:\n");
        parallel_method2(A, x, b, n, threads_number);
        for (int i = 0; i < n; i++) {
            x[i] = 0;
        }
    }
    return 0;
}
