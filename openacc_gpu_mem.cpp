#include <iostream>
#include <vector>
#include <cstdlib>
#include <chrono>
#include <algorithm>  // for std::min

using namespace std;
#define N 1000
#define TILE_SIZE 32
#define IDX(i, j) ((i)*N + (j))

int main() {
    // Allocate contiguous memory for matrices as 1D vectors.
    vector<float> A(N * N);
    vector<float> B(N * N);
    vector<float> C(N * N, 0);

    srand(time(nullptr));
    // Initialize matrices A and B.
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[IDX(i, j)] = rand() % 10;
            B[IDX(i, j)] = rand() % 10;
        }
    }

    // Obtain raw pointers for explicit data management.
    float* a = A.data();
    float* b = B.data();
    float* c = C.data();

    auto start = chrono::high_resolution_clock::now();

    // Use an OpenACC data region that:
    // - Creates 'c' on the device persistently (avoiding repeated transfers).
    // - Copies in 'a' and 'b' once.
    #pragma acc data create(c[0:N*N]) copyin(a[0:N*N], b[0:N*N])
    {
        // Perform tiled matrix multiplication with loop unrolling.
        #pragma acc parallel loop gang collapse(2)
        for (int i = 0; i < N; i += TILE_SIZE) {
            for (int j = 0; j < N; j += TILE_SIZE) {
                for (int ii = i; ii < min(i + TILE_SIZE, N); ii++) {
                    for (int jj = j; jj < min(j + TILE_SIZE, N); jj++) {
                        float sum = 0.0f;
                        // Unroll the innermost loop by a factor of 4.
                        #pragma acc loop vector
                        for (int k = 0; k < N; k += 4) {
                            sum += a[IDX(ii, k)]   * b[IDX(k,   jj)];
                            sum += a[IDX(ii, k+1)] * b[IDX(k+1, jj)];
                            sum += a[IDX(ii, k+2)] * b[IDX(k+2, jj)];
                            sum += a[IDX(ii, k+3)] * b[IDX(k+3, jj)];
                        }
                        c[IDX(ii, jj)] = sum;
                    }
                }
            }
        }
        // Update the host memory with the computed result from the device.
        #pragma acc update self(c[0:N*N])
    }

    auto end = chrono::high_resolution_clock::now();
    double seconds = chrono::duration<double>(end - start).count();
    cout << "Optimized OpenACC GPU with Memory Execution Time: " << seconds << " seconds." << endl;

    return 0;
}
