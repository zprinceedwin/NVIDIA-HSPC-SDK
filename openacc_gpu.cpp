#include <iostream>
#include <vector>
#include <cstdlib>
#include <chrono>

using namespace std;

#define N 1000
#define TILE_SIZE 32
#define IDX(i, j) ((i) * N + (j))

int main() {
    vector<float> A(N * N);
    vector<float> B(N * N);
    vector<float> C(N * N, 0);

    srand(time(nullptr));

    // Initialize matrices
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            A[IDX(i, j)] = rand() % 10;
            B[IDX(i, j)] = rand() % 10;
        }

    float* a = A.data();
    float* b = B.data();
    float* c = C.data();

    auto start = chrono::high_resolution_clock::now();

#pragma acc data copyin(a[0:N*N], b[0:N*N]) copyout(c[0:N*N])
    {
#pragma acc parallel loop gang collapse(2)
        for (int i = 0; i < N; i += TILE_SIZE) {
            for (int j = 0; j < N; j += TILE_SIZE) {
                for (int ii = i; ii < i + TILE_SIZE; ii++) {
                    for (int jj = j; jj < j + TILE_SIZE; jj++) {
                        float sum = 0.0f;
#pragma acc loop vector
                        for (int k = 0; k < N; k++) {
                            sum += a[IDX(ii, k)] * b[IDX(k, jj)];
                        }
                        c[IDX(ii, jj)] = sum;
                    }
                }
            }
        }
    }

    auto end = chrono::high_resolution_clock::now();
    double seconds = chrono::duration<double>(end - start).count();
    cout << "Optimized OpenACC GPU Execution Time: " << seconds << " seconds." << endl;

    return 0;
}
