#include <iostream>
#include <vector>
#include <cstdlib>
#include <chrono>

using namespace std;

#define N 1000
#define IDX(i, j) ((i) * N + (j))

int main() {
    vector<vector<float>> A(N, vector<float>(N));
    vector<vector<float>> B(N, vector<float>(N));
    vector<vector<float>> C(N, vector<float>(N, 0));

    srand(time(nullptr));

    // Initialize matrices
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            A[i][j] = rand() % 10;
            B[i][j] = rand() % 10;
        }

    auto start = chrono::high_resolution_clock::now();

    // OpenACC CPU Parallelization
#pragma acc parallel loop collapse(2)
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            float temp = A[i][k];
            for (int j = 0; j < N; j++) {
                C[i][j] += temp * B[k][j];
            }
        }
    }

    auto end = chrono::high_resolution_clock::now();
    double seconds = chrono::duration<double>(end - start).count();
    cout << "Optimized OpenACC CPU Execution Time: " << seconds << " seconds." << endl;

    return 0;
}
