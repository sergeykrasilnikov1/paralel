/*
 * Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <fstream>
#include <iomanip>
#include "laplace2d.hpp"

#define OFFSET(x, y, m) (((x)*(m)) + (y))



Laplace::Laplace(int m, int n) : m(m), n(n){
    A = (double *)malloc(sizeof(double) * n * m);
    Anew = (double *)malloc(sizeof(double) * n * m);
}

Laplace::~Laplace() {
    free(A);
    free(Anew);
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
    for (int i=2;i<n-2;i++) {

        A[OFFSET(1,i,n)] = heat_points[0].second + (heat_points[1].second - heat_points[0].second) * i / (n - 1);

        A[OFFSET(-2,i,n)] = heat_points[3].second + (heat_points[2].second - heat_points[3].second) * i / (n - 1);

        A[OFFSET(i,1,n)] = heat_points[0].second + (heat_points[3].second - heat_points[0].second) * i / (n - 1);

        A[OFFSET(i,-2,n)] = heat_points[1].second + (heat_points[2].second - heat_points[1].second) * i / (n - 1);
    }
}

double Laplace::calcNext()
{
    double error = 0.0;
    for( int j = 1; j < n-1; j++)
    {
        for( int i = 1; i < m-1; i++ )
        {
            int points = 1;
            points += i > 1 ? 1 : 0;
            points += j > 1 ? 1 : 0;
            points += i < n - 2 ? 1 : 0;
            points += j < n - 2 ? 1 : 0;
            Anew[OFFSET(j, i, n)] = (A[OFFSET(j, i + 1, n)] + A[OFFSET(j, i - 1, n)] +
                                     A[OFFSET(j - 1, i, n)] + A[OFFSET(j + 1, i, n)] +
                                     A[OFFSET(j, i, n)])  / points;
            error = fmax( error, fabs(Anew[OFFSET(j, i, m)] - A[OFFSET(j, i , m)]));
        }
    }
    return error;
}

void Laplace::swap()
{
    for( int j = 1; j < n-1; j++)
    {
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