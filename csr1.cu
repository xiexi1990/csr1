#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cnpy.h>

using namespace std;

struct SparseMatrix {
    int n; // 矩阵维度
    vector<int> row_ptr; // 行指针
    vector<int> col_idx; // 列索引
    vector<float> values; // 非零元素

    SparseMatrix(int n) {
        this->n = n;
        row_ptr.resize(n+1);
        row_ptr[0] = 0;
    }
};

SparseMatrix generate_random_sparse_matrix(int n, int num_nonzeros) {
    SparseMatrix mat(n);

    int max_nonzeros_per_row = num_nonzeros / n;
    int remaining_nonzeros = num_nonzeros - max_nonzeros_per_row * n;

    srand(123);

    int k = 0;
    for (int i = 0; i < n; i++) {
        int num_nonzeros_in_row = max_nonzeros_per_row;
        if (remaining_nonzeros > 0) {
            num_nonzeros_in_row += rand() % 2;
            remaining_nonzeros--;
        }
        for (int j = 0; j < num_nonzeros_in_row; j++) {
            float val = (float) rand() / RAND_MAX; // 随机生成一个0到1之间的数
            int col = rand() % n; // 随机选择一个列
            mat.col_idx.push_back(col);
            mat.values.push_back(val);
            k++;
        }
        mat.row_ptr[i+1] = k;
    }

    return mat;
}

int main() {
    int n = 50; // 矩阵维度
    int num_nonzeros = 300; // 矩阵非零元素个数
    int ndim = 32;

    SparseMatrix mat = generate_random_sparse_matrix(n, num_nonzeros);


    cnpy::npz_t npz = cnpy::npz_load("npz36.npz");
    cnpy::NpyArray npy_shape = npz["shape"];
    uint32_t num_rows = npy_shape.data<uint32_t>()[0];
    uint32_t num_cols = npy_shape.data<uint32_t>()[2];
    cnpy::NpyArray npy_data = npz["data"];
    uint32_t nnz = npy_data.shape[0];
    cnpy::NpyArray npy_indices = npz["indices"];
    cnpy::NpyArray npy_indptr = npz["indptr"];
    // csr_matrix.adj_data.insert(csr_matrix.adj_data.begin(), &npy_data.data<float>()[0],
    //     &npy_data.data<float>()[nnz]);
    // csr_matrix.adj_indices.insert(csr_matrix.adj_indices.begin(), &npy_indices.data<uint32_t>()[0],
    //     &npy_indices.data<uint32_t>()[nnz]);
    // csr_matrix.adj_indptr.insert(csr_matrix.adj_indptr.begin(), &npy_indptr.data<uint32_t>()[0],
    //     &npy_indptr.data<uint32_t>()[num_rows + 1]);


    int *ptr, *idx;
    float *val, *vin, *vout;
    cudaMallocManaged(&ptr, (n+1) * sizeof(int));
    cudaMallocManaged(&idx, num_nonzeros * sizeof(int));
    cudaMallocManaged(&val, num_nonzeros * sizeof(float));
    cudaMallocManaged(&vin, n * ndim * sizeof(float));
    cudaMallocManaged(&vout, n * ndim * sizeof(float));

    memset(vout, 0, n * ndim * sizeof(float));
    memcpy(ptr, mat.row_ptr.data(), (n+1) * sizeof(int));


    cudaFree(ptr);
    cudaFree(idx);
    cudaFree(val);
    cudaFree(vin);
    cudaFree(vout);

    return 0;
}
