#include "nn.hpp"

Matrix debug_initialize_weights(int out, int in) {
    Matrix W(out, in + 1);
    int cnt = 1;
    for (int j = 0; j < in + 1; j++)
        for (int i = 0; i < out; i++)
            W(i, j) = sin(cnt++) / 10;
    return W;
}

void check_gradients(double lambda) {
    int input_layer_size = 3;
    int hidden_layer_size = 5;
    int num_labels = 3;
    int m = 5;
    vector<Matrix> Theta;
    Theta.push_back(debug_initialize_weights(hidden_layer_size, input_layer_size));
    Theta.push_back(debug_initialize_weights(num_labels, hidden_layer_size));
    NeuralNet debugnn(Theta);

    Matrix X = debug_initialize_weights(m, input_layer_size - 1);
    Matrix y(m, 1);
    for (int i = 0; i < m; i++)
        y(i, 0) = fmod(i + 1, num_labels);

    vector<Matrix> grad = debugnn.feed_forward(X, y, lambda).grad();
    vector<Matrix> numgrad = debugnn.compute_numerical_gradient(X, y, lambda);
    cout << grad[0] << endl;
    cout << numgrad[0];
}

int main()
{
    check_gradients(1);
    return 0;
}