#include <iostream>
#include <string>
#include <fstream>
#include <tuple>
#include "nn.hpp"

using namespace std;
using namespace toymat;

void write(string filename, Matrix& X) {
    ofstream myfile;
    myfile.open(filename);
    myfile << X;
    myfile.close();
}

Matrix load(string filename, size_t rows, size_t cols) {
    Matrix data(rows, cols);
    ifstream myfile (filename);
    if (myfile.is_open()) {
        for (size_t i = 0; i < rows; i++)
            for (size_t j = 0; j < cols; j++)
                myfile >> data(i, j);
        myfile.close();
    } else {
        exit(1);
    }
    return data;
}

int main() {
    Matrix X = load("mnist.dat", 5000, 400);
    Matrix y = load("y.dat", 5000, 1);
    NeuralNet nn({400, 25, 10});
    nn.training(X, y, 2.5, 0.27, 500);
    return 0;
}