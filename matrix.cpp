#include "matrix.hpp"

static int precision = 5;

namespace toymat {

Matrix::Matrix() : rows(0), cols(0), mat(nullptr) { }

Matrix::Matrix(size_t rows, size_t cols)
    : rows(rows), cols(cols), mat(new double[rows * cols] {}) { }

Matrix::Matrix(initializer_list< initializer_list<double> > mat)
    : Matrix(mat.size(), mat.begin()->size()) {
    size_t i = 0;
    for (auto row : mat)
        for (auto elem : row)
            this->mat[i++] = elem;
}

Matrix::Matrix(const Matrix& that)
    : rows(that.rows), cols(that.cols), mat(new double[rows * cols]) {
    memcpy(this->mat, that.mat, sizeof(double) * rows * cols);
}

Matrix::Matrix(Matrix&& that)
    : rows(that.rows), cols(that.cols), mat(that.mat) {
    that.rows = 0;
    that.cols = 0;
    that.mat = nullptr;
}

Matrix::~Matrix() { delete[] mat; }

Matrix& Matrix::operator=(const Matrix& that) {
    Matrix tmp(that);
    swap(tmp, *this);
    return *this;
}

Matrix& Matrix::operator=(Matrix&& that) {
    swap(rows, that.rows);
    swap(cols, that.cols);
    swap(mat, that.mat);
    return *this;
}

Matrix& Matrix::reshape(size_t rows, size_t cols) {
    assert_msg(rows * cols == size(), "\n - Error using reshape"
        "\n - To RESHAPE the number of elements must not change.");
    this->rows = rows;
    this->cols = cols;
    return *this;
}

Matrix submat(const Matrix& X, 
              int startrow, int endrow, int startcol, int endcol) {
    if (endrow == -1) endrow = X.get_row() - 1;
    if (endcol == -1) endcol = X.get_col() - 1;
    Matrix result(endrow - startrow + 1, endcol - startcol + 1);
    for (size_t i = 0; i < result.rows; i++)
        for (size_t j = 0; j < result.cols; j++)
            result(i, j) = X.mat[(startrow + i) * X.cols + (startcol + j)];
    return result;
}

Matrix submat(const Matrix& X, vector<size_t> indices) {
    size_t width = X.get_col();
    Matrix result = Matrix(indices.size(), width);

    size_t line = 0;
    for (size_t i : indices) {
        assert_msg(i >= 0 && i < X.get_row(), "\n - Error using submat"
            "\n - \'" + to_string(i) + "\' is not a valid index");
        memcpy(result.mat + line * width, X.mat + i * width, sizeof(double) * width);
        line++;
    }
    return result;
}

Matrix& operator*=(Matrix& X, const Matrix& Y) {
    assert_msg(X.cols == Y.rows, "\n - Error using *"
        "\n - Inner matrix dimensions must agree.");
    double *result = new double[X.rows * Y.cols] {};
    for (size_t i = 0; i < X.rows; i++) {
        for (size_t k = 0; k < X.cols; k++) {
            double r = X(i, k);
            for (size_t j = 0; j < Y.cols; j++) {
                result[i * Y.cols + j] += r * Y(k, j);
            }
        }
    }
    if (X.mat) delete[] X.mat;
    X.mat = result;
    X.cols = Y.cols;
    return X;
}

Matrix matcmp(const Matrix& X, const Matrix& Y) {
    assert_msg(X.rows == Y.rows && X.cols == Y.cols, "\n - Error using MATCMP"
        "\n - Matrix dimensions must agree.");
    Matrix result(X.rows, X.cols);
    for (size_t i = 0; i < X.rows; i++)
        for (size_t j = 0; j < X.cols; j++)
            result(i, j) = (abs(X(i, j) - Y(i, j)) < 1e-12);
    return result;
}

Matrix emul(const Matrix& X, const Matrix& Y) {
    assert_msg(X.rows == Y.rows && X.cols == Y.cols, "\n - Error using "
        "element-wise multiply\n - Matrix dimensions must agree.");
    Matrix result = X;
    for (size_t i = 0; i < X.size(); i++)
        result.mat[i] *= Y.mat[i];
    return result;
}

Matrix t(const Matrix& X) {
    Matrix result(X.cols, X.rows);
    for (size_t i = 0; i < X.rows; i++)
        for (size_t j = 0; j < X.cols; j++)
            result(j, i) = X(i, j);
    return result;
}

Matrix concat(const Matrix X, const Matrix Y, int dimension) {
    Matrix result;
    if (dimension == 1) {
        assert_msg(X.cols == Y.cols, "\n - Error using vertcat\n - Dimensions "
            "of matrices being concatenated are not consistent.");
        result = Matrix(X.rows + Y.rows, X.cols);
        memcpy(result.mat, X.mat, sizeof(double) * X.size());
        memcpy(result.mat + X.size(), Y.mat, sizeof(double) * Y.size());
        return result;
    } else if (dimension == 2) {
        assert_msg(X.rows == Y.rows, "\n - Error using horzcat\n - Dimensions "
            "of matrices being concatenated are not consistent.");
        result = Matrix(X.rows, X.cols + Y.cols);
        for (size_t i = 0; i < result.rows; i++) {
            memcpy(result.mat + i * result.cols,
                   X.mat + i * X.cols,
                   sizeof(double) * X.cols);
            memcpy(result.mat + i * result.cols + X.cols,
                   Y.mat + i * Y.cols,
                   sizeof(double) * Y.cols);
        }
    } else {
        assert_msg(0, "\n - Error using concatenate\n - Invalid dimension.");
    }
    return result;
}

tuple<Matrix, Matrix> max(const Matrix &X) {
    Matrix maximum = submat(X, X.rows - 1, X.rows - 1, 0, -1);
    Matrix indices = Ones(1, X.cols) * (X.rows - 1);
    for (int r = X.rows - 1; r > -1; r--) {
        for (size_t c = 0; c < X.cols; c++) {
            if (X(r, c) >= maximum(0, c)) {
                maximum(0, c) = X(r, c);
                indices(0, c) = r;
            }
        }
    }
    return make_tuple(maximum, indices);
}

Matrix sum(const Matrix& X, int dimension) {
    Matrix result;
    if (dimension == 2 || X.rows == 1) {
        result = Matrix(X.rows, 1);
        for (size_t c = 0; c < X.cols; c++)
            for (size_t r = 0; r < X.rows; r++)
                result(r, 0) += X(r, c);
    } else if (dimension == 1) {
        result = Matrix(1, X.cols);
        for (size_t r = 0; r < X.rows; r++)
            for (size_t c = 0; c < X.cols; c++)
                result(0, c) += X(r, c);
    } else {
        result = X;
    }
    return result;
}

Matrix apply(Matrix X, function<double(double)> func) {
    Matrix result(X.rows, X.cols);
    for (size_t i = 0; i < X.size(); i++)
        result.mat[i] = func(X.mat[i]);
    return result;
}

Zeros::Zeros(size_t rows, size_t cols) : Matrix(rows, cols) {
    for (size_t i = 0; i < rows * cols; i++)
        mat[i] = 0;
}

Zeros::Zeros(size_t rows) : Zeros(rows, rows) {}

Ones::Ones(size_t rows, size_t cols) : Matrix(rows, cols) {
    for (size_t i = 0; i < rows * cols; i++)
        mat[i] = 1;
}

Ones::Ones(size_t rows) : Ones(rows, rows) {}

Rand::Rand(size_t rows, size_t cols) : Matrix(rows, cols) {
    random_device generator;
    uniform_real_distribution<double> distribution(0, 1);
    for (size_t i = 0; i < rows * cols; i++)
        mat[i] = distribution(generator);
}

Rand::Rand(size_t rows) : Rand(rows, rows) {}

Eye::Eye(size_t rows, size_t cols) : Matrix(rows, cols) {
    int min = rows < cols ? rows : cols;
    for (size_t i = 0; i < min; i++)
        mat[index(i, i)] = 1;
}

Eye::Eye(size_t rows) : Eye(rows, rows) {}

ostream& operator<<(ostream& os, const Matrix& X) {
    int columns;
    if (precision == 5) columns = 7;
    else                columns = 3;

    for (size_t i = 0; i < X.cols; i += columns) {
        if (X.cols > columns) {
            if (X.cols - i > columns)
                os << "\nColumns " << i << " through " << i + columns - 1 << "\n\n";
            else if (X.cols - i == 1)
                os << "\nColumns " << i << "\n\n";
            else
                os << "\nColumns " << i << " through " << X.cols - 1 << "\n\n";
        }
        
        for (size_t j = 0; j < X.rows; j++) {
            for (size_t k = 0; k < columns && i + k < X.cols; k++)
                os << fixed << setw(precision + 6) << setprecision(precision) << X(j, i + k);
            os << "\n";
        }
    }
    return os;
}

ostream& format_long(ostream& os) {
    precision = 15;
    return os;
}

ostream& format_short(ostream& os) {
    precision = 5;
    return os;
}

} // namespace toymat