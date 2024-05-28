#include <iostream>
#include <omp.h>
#include <new>
#include <nvtx3/nvToolsExt.h>
#include <chrono>
#include <boost/program_options.hpp>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <cub/cub.cuh>
#include <algorithm>
#include <vector>

#define OFFSET(x, y, m) (((x) * (m)) + (y))


class Laplace {
private:
    double* A, * Anew;
    double* hA;
    int m, n;

public:
    Laplace(int m, int n);
    ~Laplace();
    void calcNext();
    double calcError();
    void swap();
};

__global__ void initializeKernel(double *A, double *Anew, int n, int m) {
    A[OFFSET(0,0,n)] = 10;
    A[OFFSET(0,(n-1),n)] = 20;
    A[OFFSET((n-1),0,n)] = 20;
    A[OFFSET((n-1),(n-1),n)] = 30;

    Anew[OFFSET(0,0,n)] = 10;
    Anew[OFFSET(0,n-1,n)] = 20;
    Anew[OFFSET(n-1,0,n)] = 20;
    Anew[OFFSET(n-1,n-1,n)] = 30;
    for (int i=1;i<n-1;i++) {
        A[OFFSET(0,i,n)] = 10 + (20 - 10) * i / (n - 1);
        A[OFFSET(n-1,i,n)] = 20 + (30 - 20) * i / (n - 1);
        A[OFFSET(i,0,n)] = 10 + (20 - 10) * i / (n - 1);
        A[OFFSET(i,n-1,n)] = 20 + (30 - 20) * i / (n - 1);

        Anew[OFFSET(0,i,n)] = 10 + (20 - 10) * i / (n - 1);
        Anew[OFFSET(n-1,i,n)] = 20 + (30 - 20) * i / (n - 1);
        Anew[OFFSET(i,0,n)] = 10 + (20 - 10) * i / (n - 1);
        Anew[OFFSET(i,n-1,n)] = 20 + (30 - 20) * i / (n - 1);
    }
}


Laplace::Laplace(int m, int n) : m(m), n(n){
    cudaMalloc(&A, m * n * sizeof(double));
    cudaMalloc(&Anew, m * n * sizeof(double));
    int blockSize = 256;
    int numBlocks = (n * m + blockSize - 1) / blockSize;
    initializeKernel<<<numBlocks, blockSize>>>(A, Anew, n, m);
    cudaDeviceSynchronize();
}


Laplace::~Laplace() {
    std::ofstream out("out.txt");
    out << std::fixed << std::setprecision(5);
    hA = new double[n * m];
    cudaMemcpy(hA, A, n * m * sizeof(double), cudaMemcpyDeviceToHost);
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            out << std::left << std::setw(10) << hA[OFFSET(j, i, m)] << " ";
        }
        out << std::endl;
    }
    out.close();
    cudaFree(A);
    cudaFree(Anew);
    delete (hA);
}


__global__ void calcNextKernel(double *A, double *Anew, int m, int n) {
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i < m - 1 && j < n - 1) {
        Anew[OFFSET(j, i, n)] = 0.2 * (A[OFFSET(j, i + 1, n)] + A[OFFSET(j, i - 1, n)] +
                                      A[OFFSET(j - 1, i, n)] + A[OFFSET(j + 1, i, n)] +
                                      A[OFFSET(j, i, n)]);
    }
}

void Laplace::calcNext() {
    dim3 block_size(16, 16);
    dim3 grid_size((m + block_size.x - 2) / block_size.x, (n + block_size.y - 2) / block_size.y);

    calcNextKernel<<<grid_size, block_size>>>(A, Anew, m, n);

    cudaDeviceSynchronize();
}

__global__ void calcErrorKernel(const double* __restrict__ A, const double* __restrict__ Anew, int n, int m, double* blockMaxErrors) {
    extern __shared__ double sharedData[];
    int tid = threadIdx.x;

    double localMax = 0.0;

    for (int j = blockIdx.y + 1; j < n - 1; j += gridDim.y) {
        for (int i = threadIdx.x + 1; i < m - 1; i += blockDim.x) {
            double error = fabs(Anew[OFFSET(j, i, m)] - A[OFFSET(j, i, m)]);
            localMax = fmax(localMax, error);
        }
    }


    sharedData[tid] = localMax;
    __syncthreads();

    typedef cub::BlockReduce<double, 1024> BlockReduce; 
    __shared__ typename BlockReduce::TempStorage temp_storage;

    double blockMax = BlockReduce(temp_storage).Reduce(sharedData[tid], cub::Max());


    if (tid == 0) {
        blockMaxErrors[blockIdx.x + blockIdx.y * gridDim.x] = blockMax;
    }
}

double Laplace::calcError() {
    int numBlocks = (m - 2 + 1023) / 1024;
    dim3 blocks(numBlocks, (n - 2 + numBlocks - 1) / numBlocks);
    dim3 threads(1024);


    double* d_blockMaxErrors;
    cudaMalloc(&d_blockMaxErrors, blocks.x * blocks.y * sizeof(double));


    size_t sharedMemSize = 1024 * sizeof(double); // Assuming block size of 1024
    calcErrorKernel<<<blocks, threads, sharedMemSize>>>(A, Anew, n, m, d_blockMaxErrors);


    double* h_blockMaxErrors = new double[blocks.x * blocks.y];
    cudaMemcpy(h_blockMaxErrors, d_blockMaxErrors, blocks.x * blocks.y * sizeof(double), cudaMemcpyDeviceToHost);


    double maxError = 0.0;
    for (int i = 0; i < blocks.x * blocks.y; ++i) {
        maxError = fmax(maxError, h_blockMaxErrors[i]);
    }


    delete[] h_blockMaxErrors;
    cudaFree(d_blockMaxErrors);

    return maxError;
}

void Laplace::swap() {
    double *d_temp;
    cudaMalloc(&d_temp, n * m * sizeof(double));
    cudaMemcpy(d_temp, A, n * m * sizeof(double), cudaMemcpyDeviceToDevice);
    double *temp = A;
    A = Anew;
    Anew = temp;
    cudaFree(d_temp);
}

namespace po = boost::program_options;

int main(int argc, char **argv) {
    int m = 4096;
    int iter_max = 1000;
    double tol = 1.0e-6;

    double error = 1.0;

    po::options_description desc("Allowed options");
    desc.add_options()
            ("help", "Usage: set precision by running -p 123, set grid size by using -n 123 and set maximum number of iterations by -i 123")
            ("precision,p", po::value<double>(&tol)->default_value(1.0e-6), "precision")
            ("grid_size,n", po::value<int>(&m)->default_value(4096), "grid size")
            ("iterations,i", po::value<int>(&iter_max)->default_value(1000), "number of iterations");
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 1;
    }


    nvtxRangePushA("init");
    Laplace a(m, m);
    nvtxRangePop();
    printf("Jacobi relaxation Calculation: %d x %d mesh\n", m, m);

    auto start = std::chrono::high_resolution_clock::now();
    int iter = 0;

    nvtxRangePushA("while");
    while (error > tol && iter < iter_max)
    {
        nvtxRangePushA("calc");
        a.calcNext();
        nvtxRangePop();

        if (iter % 1000 == 0){
            nvtxRangePushA("error");
            error = a.calcError();
            nvtxRangePop();
            printf("%5d, %0.6f\n", iter, error);
        }

        nvtxRangePushA("swap");
        a.swap();
        nvtxRangePop();

        iter++;
    }
    nvtxRangePop();

    auto end = std::chrono::high_resolution_clock::now();
    auto duration_sec = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);



    std::cout << " total: " << duration_sec.count() << "s\n";

    return 0;
}
