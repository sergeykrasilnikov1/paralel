#include <vector>
#include <cublas_v2.h>
#include <cuda_runtime.h>


class Laplace {
private:
    double* A, * Anew;
    int m, n;

public:
    Laplace(int m, int n);
    ~Laplace();
    void initialize(std::vector<std::pair<int, double>> heat_points);
    double calcNext(cublasHandle_t handle);
    void swap();
    void draw_field(int size);
};