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

    // Fully optimized loop ordering for better cache efficiency
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) { // Switch order to improve cache locality
            float temp = A[i][k]; // Store value to avoid repeated memory access
            for (int j = 0; j < N; j++) {
                C[i][j] += temp * B[k][j];
            }
        }
    }

    auto end = chrono::high_resolution_clock::now();
    double seconds = chrono::duration<double>(end - start).count();
    cout << "Optimized Sequential Execution Time: " << seconds << " seconds." << endl;

    return 0;
}
