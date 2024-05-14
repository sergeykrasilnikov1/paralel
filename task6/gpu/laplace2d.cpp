
#include <cmath>
#include <cstring>
#include <iostream>
#include <iomanip>
#include "laplace2d.hpp"
#include <vector>

#define OFFSET(x, y, m) (((x) * (m)) + (y))

Laplace::Laplace(int m, int n) : m(m), n(n)
{
    A = new double[n * m];
    Anew = new double[n * m];
}

Laplace::~Laplace()
{
#pragma acc exit data delete (this)

    delete (A);
    delete (Anew);
}

void Laplace::initialize(std::vector<std::pair<int, double>> heat_points)
{
    memset(A, 0, (n * n) * sizeof(double));
    memset(Anew, 0, n * n * sizeof(double));
    for (auto heat_point : heat_points) {
        int index = heat_point.first;
        double temp = heat_point.second;
        A[index] = temp;
        Anew[index] = temp;
    }
#pragma acc enter data copyin(this)
#pragma acc enter data copyin(A[ : n * n], Anew[ : n * n])
}

double Laplace::calcNext()
{
    double error = 0.0;
#pragma acc parallel loop reduction(max : error) present(A, Anew)
    for (int j = 1; j < n - 1; j++) {
#pragma acc loop
        for (int i = 1; i < n - 1; i++) {
            int points = 1;
            points += i > 1 ? 1 : 0;
            points += j > 1 ? 1 : 0;
            points += i < n - 2 ? 1 : 0;
            points += j < n - 2 ? 1 : 0;
            Anew[OFFSET(j, i, n)] = (A[OFFSET(j, i + 1, n)] + A[OFFSET(j, i - 1, n)] +
                                     A[OFFSET(j - 1, i, n)] + A[OFFSET(j + 1, i, n)] +
                                     A[OFFSET(j, i, n)])  / points;
            error = fmax(error, fabs(Anew[OFFSET(j, i, n)] - A[OFFSET(j, i, n)]));

        }
    }
    return error;
}

void Laplace::swap()
{
    double* temp = A;
    A = Anew;
    Anew = temp;
#pragma acc data present(A, Anew)

}

void Laplace::draw_field(int size) {
    #pragma acc exit data copyout(A[:m*n], Anew[:m*n])
    for (int i = 1; i < size - 1; i++) {
        for (int j = 1; j < size - 1; j++)
            std::cout << A[OFFSET(i, j, size)] << " ";
        std::cout << "\n";
    }
}