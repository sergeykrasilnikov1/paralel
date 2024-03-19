#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <unordered_map>
#include <cmath>
#include <fstream>
#include <random>
#include <chrono>
#include "functional"


double sin_task(double arg) {
    return std::sin(arg);
}

double sqrt_task(double arg) {
    return std::sqrt(arg);
}

double pow_task(double base, double exponent) {
    return std::pow(base, exponent);
}

template<typename TaskType>
class Server {
public:
    Server() : stop_flag(false), next_task_id(0) {}

    void start() {
        server_thread = std::thread(&Server::run, this);
    }

    void stop() {
        stop_flag = true;
        cv.notify_all();
        server_thread.join();
    }

    size_t add_task(TaskType task) {
        std::unique_lock<std::mutex> lock(mutex_tasks);
        tasks.emplace(next_task_id, std::move(task));
        cv.notify_one();
        return next_task_id++;
    }

    double request_result(size_t id_res) {
        std::unique_lock<std::mutex> lock(mutex_results);
        cv_results.wait(lock, [this, id_res]() { return results.find(id_res) != results.end(); });
        return results[id_res];
    }

private:
    void run() {
        while (!stop_flag) {
            std::unique_lock<std::mutex> lock(mutex_tasks);
            cv.wait(lock, [this]() { return !tasks.empty() || stop_flag; });

            if (!tasks.empty()) {
                auto task = std::move(tasks.front().second);
                auto task_id = tasks.front().first;
                tasks.pop();
                lock.unlock();

                double result = task();

                std::unique_lock<std::mutex> lock_res(mutex_results);
                results[task_id] = result;
                cv_results.notify_all();
                std::cout << result << '\n';
            }
        }
    }

private:
    std::thread server_thread;
    std::queue<std::pair<size_t, TaskType>> tasks;
    std::unordered_map<size_t, double> results;
    std::mutex mutex_tasks;
    std::mutex mutex_results;
    std::condition_variable cv;
    std::condition_variable cv_results;
    bool stop_flag;
    size_t next_task_id;
};

void client_task(Server<std::function<double()>>& server, std::string filename, int num_tasks, std::function<double()> task_generator) {
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    for (int i = 0; i < num_tasks; ++i) {
        size_t task_id = server.add_task(task_generator);
        double result = server.request_result(task_id);
        outfile << "Task " << task_id << ": " << result << std::endl;
    }

    outfile.close();
}

int main() {
    Server<std::function<double()>> server;

    server.start();

    int num_tasks_per_client = 10;
    std::thread client1(client_task, std::ref(server), "client1.txt", num_tasks_per_client, [](){ return sin_task(1.0); });
    std::thread client2(client_task, std::ref(server), "client2.txt", num_tasks_per_client, [](){ return sqrt_task(4.0); });
    std::thread client3(client_task, std::ref(server), "client3txt", num_tasks_per_client, [](){ return pow_task(3.0, 2.0); });

    client1.join();
    client2.join();
    client3.join();

    server.stop();

    return 0;
}
