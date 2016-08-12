#include "nn.hpp"

NeuralNet::NeuralNet(vector<int> layers) : J(-1) {
    for (int i = 1; i < layers.size(); i++)
        Theta.push_back(random_init_weights(layers[i], layers[i - 1]));
}

NeuralNet::NeuralNet(vector<Matrix> Theta) : J(-1), Theta(Theta) { }

void NeuralNet::training(const Matrix& X, const Matrix& y, double learning_rate, double lambda, int epoch) {
    vector<size_t> sel(X.get_row());
    size_t nextValue = 0;
    generate(sel.begin(), sel.end(), [&]{ return nextValue++; });
    random_device generator;
    shuffle(sel.begin(), sel.end(), generator);

    const int NUM_TRAINING = 0.7 * X.get_row();
    vector<size_t> training = vector<size_t>(sel.begin(), sel.begin() + NUM_TRAINING);
    vector<size_t> test = vector<size_t>(sel.begin() + NUM_TRAINING, sel.end());

    Matrix training_set   = submat(X, training);
    Matrix test_set       = submat(X, test);
    Matrix training_label = submat(y, training);
    Matrix test_label     = submat(y, test);

    for (int i = 0; i < epoch; i++) {
        vector<Matrix> Theta_grad = feed_forward(training_set, training_label, lambda).grad();
        for (int j = 0; j < Theta.size(); j++)
            Theta[j] -= Theta_grad[j] * learning_rate;

        cout << "Epoch" << setw(4) << i;
        cout << "  |  Cost: "           << setw(8) << setprecision(5) << J;
        cout << "  |  Training error: " << setw(8) << setprecision(5) << eval_error(training_set, training_label);
        cout << "  |  Test error: "     << setw(8) << setprecision(5) << eval_error(test_set, test_label);
        cout << endl;
    }
}

const NeuralNet& NeuralNet::feed_forward(const Matrix& X, const Matrix& y, int lambda) {
    int num_labels = Theta.back().get_row();
    int m = X.get_row();        // num of training examples
    vector<Matrix> z, a;        // logistics and activations of all layers

    // feed-forward
    Matrix activation = t(X);   // input layer
    for (auto theta : Theta) {
        a.push_back(concat(Ones(1, m), activation));
        Matrix z_i = theta * a.back();
        z.push_back(z_i);
        activation = apply(z_i, [](double x) { return sigmoid(x); });
    }
    a.push_back(activation);

    // expand y to matrix y_expand where every column is a training example
    Matrix y_expand = Zeros(num_labels, m);
    for (size_t i = 0; i < m; i++)
        y_expand(y(i, 0), i) = 1;
    
    // total costs without regularization
    double Jsum = sum(sum(
        emul(y_expand, apply(a.back(), [](double x) { return -log(x); }))
      + emul(apply(y_expand, [](double x) { return x - 1; }), apply(a.back(), [](double x) { return log(1 - x); }))))
    .toScalar();

    double reg_sum = 0;         // sum of regularization terms
    for (auto theta : Theta)
        reg_sum += sum(sum(apply(submat(theta, 0, -1, 1, -1), [](double x) { return x * x; }))).toScalar();

    // cost function with regularization
    J = Jsum / m + reg_sum * lambda / 2 / m;

    Matrix delta;
    Theta_grad.erase(Theta_grad.begin(), Theta_grad.end());
    for (int i = Theta.size() - 1; i >= 0; i--) {
        if (i == Theta.size() - 1)
            delta = a.back() - y_expand;
        else
            delta = emul(submat(t(Theta[i + 1]) * delta, 1, -1, 0, -1), apply(z[i], [](double x) { return sigmoid_prime(x); }));
        Theta_grad.push_back(delta * t(a[i]) / m + 1.0 * lambda / m * concat(Zeros(Theta[i].get_row(), 1), submat(Theta[i], 0, -1, 1, -1), 2));
    }
    reverse(Theta_grad.begin(), Theta_grad.end());
    return *this;
}

tuple<Matrix, Matrix> NeuralNet::predict(const Matrix& X) const {
    Matrix activation = t(X);
    for (auto theta : Theta)
        activation = apply(theta * concat(Ones(1, X.get_row()), activation), [](double x) { return sigmoid(x); });
    return max(activation);
}

double NeuralNet::eval_error(const Matrix& X, const Matrix& y) const {
    Matrix prob, digits;
    tie(prob, digits) = predict(X);
    return 1 - sum(matcmp(t(digits), y)).toScalar() / X.get_row();
}

double NeuralNet::cost() const {
    if (J < 0) throw runtime_error("No feedforward.\n");
    return J;
}

vector<Matrix> NeuralNet::grad() const {
    if (J < 0) throw runtime_error("No feedforward.\n");
    return Theta_grad;
}

Matrix NeuralNet::random_init_weights(int in, int out) {
    double epsilon_init = 0.12;
    return apply(Rand(in, 1 + out), [&](double rdm) { return rdm * 2 * epsilon_init - epsilon_init; });
}

vector<Matrix> NeuralNet::compute_numerical_gradient(const Matrix& X, const Matrix& y, int lambda) {
    vector<Matrix> numgrad;
    double epsilon = 1e-4;
    for (int k = 0; k < Theta.size(); k++) {
        int row = Theta[k].get_row();
        int col = Theta[k].get_col();
        Matrix grad(row, col);
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                Theta[k](i, j) += epsilon;
                double loss1 = feed_forward(X, y, lambda).cost();
                Theta[k](i, j) -= 2 * epsilon;
                double loss2 = feed_forward(X, y, lambda).cost();
                Theta[k](i, j) += epsilon;
                // Compute Numerical Gradient
                grad(i, j) = (loss1 - loss2) / (2 * epsilon);
            }
        }
        numgrad.push_back(grad);
    }
    return numgrad;
}
