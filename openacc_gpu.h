#include <iostream>
#include <vector>
#include <cstdlib>
#include <chrono>

using namespace std;

#define N 1000

int main() {
    vector<vector<float>> A(N, vector<float>(N));
    vector<vector<float>> B(N, vector<float>(N));
    vector<vector<float>> C(N, vector<float>(N, 0));

    srand(time(nullptr));
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            A[i][j] = rand() % 10;
            B[i][j] = rand() % 10;
        }

    auto start = chrono::high_resolution_clock::now();

#pragma acc kernels
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            for (int k = 0; k < N; ++k)
                C[i][j] += A[i][k] * B[k][j];

    auto end = chrono::high_resolution_clock::now();
    auto micros = chrono::duration_cast<chrono::microseconds>(end - start).count();
    cout << "OpenACC (GPU) Execution Time: " << micros << " microseconds." << endl;

    return 0;
}
