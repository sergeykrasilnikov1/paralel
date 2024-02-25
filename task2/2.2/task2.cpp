#include <stdio.h>
#include <omp.h>
#include <time.h>
#include <inttypes.h>
#include <stdlib.h>
#include <math.h>

const double PI = 3.14159265358979323846;
const double a = -4.0;
const double b = 4.0;
const int nsteps = 40000000;

double cpuSecond()
{
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ((double)ts.tv_sec + (double)ts.tv_nsec * 1.e-9);
}

double func(double x)
{
    return exp(-x * x);
}

double integrate(double (*func)(double), double a, double b, int n)
{
    double h = (b - a) / n;
    double sum = 0.0;

    for (int i = 0; i < n; i++)
        sum += func(a + h * (i + 0.5));

    sum *= h;

    return sum;
}

double integrate_omp(double (*func)(double), double a, double b, int n, int n_threads)
{
    double h = (b - a) / n;
    double sum = 0.0;
#pragma omp parallel num_threads(n_threads)
    {
        double sumloc = 0.0;
#pragma omp for
        for (int i = 0; i < n; i++)
            sumloc += func(a + h * (i + 0.5));
#pragma omp atomic
        sum += sumloc;
    }
    sum *= h;
    return sum;
}
double run_serial()
{
    double t = cpuSecond();
    double res = integrate(func, a, b, nsteps);
    t = cpuSecond() - t;
    printf("Result (serial): %.12f; error %.12f\n", res, fabs(res - sqrt(PI)));
    return t;
}
double run_parallel(int n_threads)
{
    double t = cpuSecond();
    double res = integrate_omp(func, a, b, nsteps, n_threads);
    t = cpuSecond() - t;
    printf("Result (parallel): %.12f; error %.12f\n", res, fabs(res - sqrt(PI)));
    return t;
}

void tests() {
    double serial_time = run_serial();
    printf("Elapsed time(serial realization)  %lf sec.\n", serial_time);
    int threads[] = { 2, 4, 7, 8, 16, 20, 40 };
    for (int i = 0; i < 7; i++) {
        printf("test with %d threads\n", threads[i]);
        double parallel_time = run_parallel(threads[i]);
        printf("Elapsed time(parallel realization)  %lf sec.\n", parallel_time);
        printf("Speedup  %lf\n", serial_time/parallel_time);
    }
}




int main(int argc, char **argv) {
    printf("Integration f(x) on [%.12f, %.12f], nsteps = %d\n", a, b, nsteps);
    tests();
    return 0;
}
