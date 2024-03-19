#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <chrono>

std::mutex mtx;



void init(std::vector<double>& a, std::vector<double>& b, int start, int end) {
    for (int i = start; i < end; i++) {
        b[i] = i;
        for (int j = 0; j < b.size(); j++) {
            a[i * b.size() + j] = i + j;
        }
    }
}

void matrixVectorMultiply(const std::vector<double>& matrix, const std::vector<double>& vector, std::vector<double>& result, int n, int start, int end) {
    for (int i = start; i < end; ++i) {
        double sum = 0;
        for (int j = 0; j < n; ++j) {
            sum += matrix[i * n + j] * vector[j];
        }
        mtx.lock();
        result[i] = sum;
        mtx.unlock();
    }
}

int main() {
    for (int z = 1; z <= 2; z++) {
        int n = 20000 * z;
        int m = 20000 * z;
        std::cout << n << "x" << m << " test\n";
        std::vector<int> threads_list = { 2, 4, 7, 8, 16, 20, 40 };
        std::vector<double> a, b, c;
        a.resize(n * m);
        b.resize(n);
        c.resize(n);
        std::cout << "Serial test\n";
        const auto start_serial = std::chrono::steady_clock::now();
        init(a, b, 0, n);
        matrixVectorMultiply(a, b, c, n,0, n);
        const auto end_serial = std::chrono::steady_clock::now();
        const std::chrono::duration<double> elapsed_seconds_serial = end_serial - start_serial;
        double serial_time = elapsed_seconds_serial.count();
        std::cout << "Elapsed time: " << serial_time << "\n";
        std::cout << "Parallel test\n";
        for (auto thread_num : threads_list) {
            std::cout << thread_num << " threads\n";
            const auto start_parallel = std::chrono::steady_clock::now();
            std::vector<std::thread> initThreads;
            for (int i = 0; i < thread_num; ++i) {
                int start = i * (n / thread_num);
                int end = (i + 1) * (n / thread_num);
                initThreads.emplace_back(init, std::ref(a), std::ref(b), start, end);
            }

            // Ожидание завершения потоков
            for (auto& t : initThreads) {
                t.join();
            }
            std::vector<std::thread> multiplyThreads;
            for (int i = 0; i < thread_num; ++i) {
                int start = i * (n / thread_num);
                int end = (i + 1) * (n / thread_num);
                multiplyThreads.emplace_back(matrixVectorMultiply, std::cref(a), std::cref(b), std::ref(c), n, start, end);
            }

            // Ожидание завершения потоков
            for (auto& t : multiplyThreads) {
                t.join();
            }
            const auto end_parallel = std::chrono::steady_clock::now();
            const std::chrono::duration<double> elapsed_seconds_parallel = end_parallel - start_parallel;
            double parallel_time = elapsed_seconds_parallel.count();
            std::cout << "Elapsed time: " << parallel_time << "\n";
            std::cout << "Speed-up: " << serial_time / parallel_time << "\n";
        }
    }
    return 0;
}
