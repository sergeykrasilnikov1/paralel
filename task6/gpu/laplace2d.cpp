
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
    memset(A, 0, n * n * sizeof(double));
    memset(Anew, 0, n * n * sizeof(double));

    for (auto heat_point : heat_points) {
        int index = heat_point.first;
        double temp = heat_point.second;
        A[index] = temp;
        Anew[index] = temp;
    }

    for (int i=2;i<n-2;i++) {

        A[OFFSET(1,i,n)] = heat_points[0].second + (heat_points[1].second - heat_points[0].second) * i / (n - 1);

        A[OFFSET(-2,i,n)] = heat_points[3].second + (heat_points[2].second - heat_points[3].second) * i / (n - 1);

        A[OFFSET(i,1,n)] = heat_points[0].second + (heat_points[3].second - heat_points[0].second) * i / (n - 1);

        A[OFFSET(i,-2,n)] = heat_points[1].second + (heat_points[2].second - heat_points[1].second) * i / (n - 1);
    }
#pragma acc enter data copyin(this)
#pragma acc enter data copyin(A[ : n * n], Anew[ : n * n])
}

double Laplace::calcNext()
{
    double error = 0.0;
#pragma acc parallel loop reduction(max : error) present(A, Anew) async
    for (int j = 1; j < n - 1; j++) {
#pragma acc loop
        for (int i = 1; i < n - 1; i++) {
            int points = 5;
            if (i == 1 || i == m - 2) points--;
            if (j == 1 || j == n - 2) points--;
            Anew[OFFSET(j, i, n)] = (A[OFFSET(j, i + 1, n)] + A[OFFSET(j, i - 1, n)] +
                                     A[OFFSET(j - 1, i, n)] + A[OFFSET(j + 1, i, n)] +
                                     A[OFFSET(j, i, n)])  / points;
            error = fmax(error, fabs(Anew[OFFSET(j, i, n)] - A[OFFSET(j, i, n)]));

        }
    }
    #pragma acc wait
    return error;

}

void Laplace::swap()
{
#pragma acc parallel loop present(A,Anew)
    for( int j = 1; j < n-1; j++)
    {
#pragma acc loop
        for( int i = 1; i < m-1; i++ )
        {
            A[OFFSET(j, i, m)] = Anew[OFFSET(j, i, m)];
        }
    }

}

void Laplace::draw_field(int size) {
    for (int i = 1; i < size - 1; i++) {
        for (int j = 1; j < size - 1; j++)
            std::cout << A[OFFSET(i, j, size)] << " ";
        std::cout << "\n";
    }
}
