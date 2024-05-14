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

#include <string.h>
#include <stdio.h>
#include <cstdlib>
#include "laplace2d.hpp"
#include <nvtx3/nvToolsExt.h>
#include <boost/program_options.hpp>

#include <chrono>
#include <iostream>

namespace po = boost::program_options;

int main(int argc, char **argv)
{
    int n = 1024;
    double err = 1.0e-6;
    int iter_max = 1000000;

    po::options_description desc("Allowed options");
    desc.add_options()
            ("n", po::value<int>(&n))
            ("iter", po::value<int>(&iter_max))
            ("err", po::value<double>(&err))
            ("draw", "Draw output matrix");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);



    double error = 1.0;


    n+=2;
    Laplace a(n, n);
    nvtxRangePushA("init");
    std::vector<std::pair<int, double>> heat_points({std::make_pair(n + 1, 10),
                                                      std::make_pair(2 * n - 2 , 20),
                                                      std::make_pair(n * n - 2 * n + 1, 30),
                                                      std::make_pair(n * n - n - 2, 40)
                                                     });

    a.initialize(heat_points);
    nvtxRangePop();
    printf("Jacobi relaxation Calculation: %d x %d mesh\n", n-2, n-2);

    auto start = std::chrono::high_resolution_clock::now();
    int iter = 0;

    nvtxRangePushA("while");
    while (error > err && iter < iter_max)
    {
        nvtxRangePushA("calc");
        error = a.calcNext();
        nvtxRangePop();

        nvtxRangePushA("swap");
        a.swap();
        nvtxRangePop();
//        if (iter % 100 == 0)
//            printf("%5d, %0.6f\n", iter, error);

        iter++;
    }
    nvtxRangePop();

    auto runtime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start);
    printf("%5d, %0.6f\n", iter, error);
    std::cout << "TIME: " << runtime.count() / 1000000. <<std::endl;
    if (vm.count("draw")) {
        a.draw_field(n);
    }
    return 0;
}