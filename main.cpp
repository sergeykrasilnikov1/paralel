#include <iostream>
#include <cmath>
#include <vector>

int main() {
    const int size = 10000000;
    double sum = 0.0;

#ifdef USE_FLOAT
    std::vector<float> array(size);
    std::cout << "using float" << std::endl;
#else
    std::vector<double> array(size);
    std::cout << "using double" << std::endl;
#endif

    for (int i = 0; i < size; ++i) {
        array[i] = std::sin(static_cast<double>(i) / size * 2 * M_PI);
        sum += array[i];
    }

    std::cout << "Sum: " << sum << std::endl;

    return 0;
}
