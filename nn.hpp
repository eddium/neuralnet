#ifndef _NN_HPP_
#define _NN_HPP_

#include <cmath>
#include <algorithm>
#include <random>
#include <vector>
#include <tuple>
#include "matrix.hpp"

using namespace toymat;

class NeuralNet {
public:
    NeuralNet(vector<int> layers);
    NeuralNet(vector<Matrix> Theta);
    void training(const Matrix& X, const Matrix& y, double learning_rate, double lambda, int epoch);
    const NeuralNet& feed_forward(const Matrix& X, const Matrix& y, int lambda);
    tuple<Matrix, Matrix> predict(const Matrix& X) const;
    double cost() const;
    double eval_error(const Matrix& X, const Matrix& y) const;
    vector<Matrix> grad() const;
private:
    double J;
    vector<Matrix> Theta;
    vector<Matrix> Theta_grad;
    vector<Matrix> compute_numerical_gradient(const Matrix& X, const Matrix& y, int lambda);
    Matrix random_init_weights(int in, int out);
};

// Utility functions
inline double sigmoid(double x)       { return 1 / (1 + exp(-x)); }
inline double sigmoid_prime(double y) { return sigmoid(y) * (1.0 - sigmoid(y)); }

#endif
