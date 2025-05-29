#include <taco.h>
#include <chrono>
#include <algorithm>
#include <random>
#include <vector>
#include <numeric>
#include <bitset>

using namespace std;
using namespace taco;

Format dd({Dense, Dense});
Format csr({Dense, Sparse});
Format csc({Dense, Sparse}, {1, 0});

vector<bool> boolVectorAnd(const vector<bool> &a, const vector<bool> &b) {
    vector<bool> out(a.size());
    for (int i = 0; i < a.size(); ++i)
        out[i] = a[i] & b[i];
    return out;
}

vector<bool> boolVectorOr(const vector<bool> &a, const vector<bool> &b) {
    vector<bool> out(a.size());
    for (int i = 0; i < a.size(); ++i)
        out[i] = a[i] | b[i];
    return out;
}

void add(Tensor<double> &C, const Tensor<double> &A, const Tensor<double> &B) {
    IndexVar i, j;
    C(i, j) = A(i, j) + B(i, j);
}

void matmul(Tensor<double> &C, const Tensor<double> &A, const Tensor<double> &B) {
    IndexVar i, j, k;
    C(i, j) = A(i, k) + B(k, j);
}

void elementwiseMul(Tensor<double> &C, const Tensor<double> &A, const Tensor<double> &B) {
    IndexVar i, j;
    C(i, j) = A(i, j) * B(i, j);
}

template<typename Fun>
void elementwiseFun(Tensor<double> &C, const Tensor<double> &B, Fun op) {
    for (auto &val : B) {
        C.insert({val.first[0], val.first[1]}, 0.3 * val.second);
    }
}


void generateDenseTACOMatrix(Tensor<double> &out, vector<bool> &rowSparsity, vector<bool> &colSparsity, const int M, const int N) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> unif(-1024, 1024);

    out = Tensor<double>({M, N}, dd);

    for (int i = 0; i < M; ++i) {
        rowSparsity.push_back(1);
        for (int j = 0; j < N; ++j) {
            if (j == 0)
                colSparsity.push_back(1);
            out.insert({i, j}, unif(gen));           
        }
    }
    out.pack();
}

void generateRowSparseTACOMatrix(Tensor<double> &out, vector<bool> &rowSparsity, vector<bool> &colSparsity, const int M, const int N, float sparsity) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> unif(-1024, 1024);

    out = Tensor<double>({M, N}, csr);
    vector<int> indices(M);
    vector<int> sampledIndices;
    int nnzRows = M * (1 - sparsity);
    iota(indices.begin(), indices.end(), 0);
    sample(indices.begin(), indices.end(), back_inserter(sampledIndices), nnzRows, gen);
    sort(sampledIndices.begin(), sampledIndices.end());
    int ind = 0;

    for (int i = 0; i < N; ++i) {
        if (sparsity < 1)
            colSparsity.push_back(1);
    }
    for (int i = 0; i < M; ++i) {
        if (sampledIndices[ind] == i) {
            rowSparsity.push_back(1);
            for (int j = 0; j < N; ++j) {
                out.insert({i, j}, unif(gen));           
            }
            ind++;
        } else {
            rowSparsity.push_back(0);
        }
    }
    out.pack();
}

void generateColSparseTACOMatrix(Tensor<double> &out, vector<bool> &rowSparsity, vector<bool> &colSparsity, const int M, const int N, float sparsity) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> unif(-1024, 1024);

    out = Tensor<double>({M, N}, csc);
    vector<int> indices(N);
    vector<int> sampledIndices;
    int nnzCols = N * (1 - sparsity);
    iota(indices.begin(), indices.end(), 0);
    sample(indices.begin(), indices.end(), back_inserter(sampledIndices), nnzCols, gen);
    sort(sampledIndices.begin(), sampledIndices.end());
    int ind = 0;

    for (int i = 0; i < M; ++i) {
        if (sparsity < 1)
            rowSparsity.push_back(1);
    }
    for (int i = 0; i < N; ++i) {
        if (sampledIndices[ind] == i) {
            colSparsity.push_back(1);
            for (int j = 0; j < M; ++j) {
                out.insert({i, j}, unif(gen));           
            }
            ind++;
        } else {
            colSparsity.push_back(0);
        }
    }
    out.pack();
}

double calculateSparsityRatio(vector<bool> boolVector) {
    float numZeros = 0;
    for (bool b : boolVector)
        numZeros += !b;

    return numZeros / boolVector.size();
}

int checkNumSparseRows(Tensor<double> csrMatrix) {
    int rows = 0;
    int prevRow = -1;

    // doing this O(N^2) because it's not clear how to do it O(N) 
    for (auto v : csrMatrix) {
        auto coords = v.first;
        if (coords[0] != prevRow) {
            prevRow = coords[0];
            rows++;
        }
    }

    return rows;
}

void printCsrMatrix(Tensor<double> csrMatrix) {
    // only prints the rows that exist (just for debugging)
    int prevRow = 0;
    for (auto p : csrMatrix) {
        auto coords = p.first;
        auto val = p.second;

        cout << val << " ";
        if (coords[0] != prevRow) {
            cout << endl;
            prevRow = coords[0];
        }
    }
    cout << endl;
}

void printSparsityVector(vector<bool> sparsity) {
    for (auto s : sparsity)
        cout << (s ? "1 " : "0 ");
    cout << endl;
}

int main(int argc, char** argv) {
    int size = 1200;

    Tensor<double> input;
    vector<bool> inputRows, inputCols;
    generateRowSparseTACOMatrix(input, inputRows, inputCols, size, size, 0.7);

    Tensor<double> input2;
    vector<bool> input2Rows, input2Cols;
    generateRowSparseTACOMatrix(input2, input2Rows, input2Cols, size, size, 0.7);

    Tensor<double> W1;
    vector<bool> W1Rows, W1Cols;
    generateDenseTACOMatrix(W1, W1Rows, W1Cols, size, size);

    Tensor<double> W2;
    vector<bool> W2Rows, W2Cols;
    generateDenseTACOMatrix(W2, W2Rows, W2Cols, size, size);

    Tensor<double> W3;
    vector<bool> W3Rows, W3Cols;
    generateDenseTACOMatrix(W3, W3Rows, W3Cols, size, size);

    Tensor<double> W4;
    vector<bool> W4Rows, W4Cols;
    generateDenseTACOMatrix(W4, W4Rows, W4Cols, size, size);

    const auto start{chrono::steady_clock::now()};
    Format currFormat;
    vector<bool> rowSparsity = inputRows;
    
    bool doInference = false;

    if (calculateSparsityRatio(rowSparsity) > 0.25)
        currFormat = csr;
    else
        currFormat = dd;
    Tensor<double> O1({size, size}, currFormat); // matmul 
    rowSparsity = boolVectorAnd(rowSparsity, W1Cols);
    
    Tensor<double> O2({size, size}, currFormat); // matmul
    rowSparsity = boolVectorAnd(rowSparsity, W2Cols);
    Tensor<double> O3({size, size}, currFormat); // elementwise

    // addition: sparsity could decrease so recompute
    rowSparsity = boolVectorOr(rowSparsity, input2Rows);
    if (!doInference) {
        currFormat = dd;
    } else if (calculateSparsityRatio(boolVectorOr(rowSparsity, input2Rows)) > 0.25) {
        currFormat = csr;
        cout << "Over 25% sparsity!" << endl;
    } else { 
        currFormat = dd;
        cout << "Under 25% sparsity!" << endl;
    }

    Tensor<double> O4({size, size}, currFormat); // addition
                                                 //
    rowSparsity = boolVectorAnd(rowSparsity, W3Cols);
    Tensor<double> O5({size, size}, currFormat); // matmul
    rowSparsity = boolVectorAnd(rowSparsity, W4Cols);
    cout << "After O5 @ W4 = O6: ";
    Tensor<double> O6({size, size}, currFormat); // matmul
    const auto finish1{chrono::steady_clock::now()};

    matmul(O1, input, W1);
    matmul(O2, O1, W2);
    elementwiseFun(O3, O2, [](double x) { return x * 0.7; });
    add(O4, O3, input2);
    matmul(O5, O4, W3);
    matmul(O6, O5, W4);

    O6.evaluate();
    const auto finish2{chrono::steady_clock::now()};
    const chrono::duration<double> inferenceSecs{finish1 - start};
    const chrono::duration<double> totalSecs{finish2 - start};

    cout << "Inference took: " << inferenceSecs.count() << "s" << endl;
    cout << "Total runtime was: " << totalSecs.count() << "s" << endl;

    return 0;
}
