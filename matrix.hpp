#ifndef _MATRIX_HPP_
#define _MATRIX_HPP_

#include <iostream>
#include <cstring>
#include <string>
#include <functional>
#include <stdexcept>
#include <random>
#include <iomanip>
#include <initializer_list>
#include <tuple>

using namespace std;

#define assert_msg(E,M) if (!(E)) throw runtime_error(M)

namespace toymat {
class Matrix {
private:
    size_t rows, cols;

protected:
    double* mat;
    inline size_t index(size_t r, size_t c) const { return r * cols + c; }

public:
    // Initialize an empty matrix
    Matrix();
    // Initialize an empty matrix with fixed height and width
    Matrix(size_t rows, size_t cols);
    // Initialize a two-dimensional matrix by a nested `initializer_list`
    Matrix(initializer_list< initializer_list<double> > mat);
    Matrix(const Matrix& that);
    Matrix(Matrix&& that);
    inline size_t size() const { return rows * cols; }
    inline size_t get_row() const { return rows; }
    inline size_t get_col() const { return cols; }
    inline double& operator()(size_t r = 1, size_t c = 1) const {
        return mat[index(r, c)];
    }
    inline double toScalar() { return mat[0]; };
    Matrix& operator=(const Matrix& that);
    Matrix& operator=(Matrix&& that);
    Matrix& reshape(size_t rows, size_t cols);
    ~Matrix();
    friend ostream& operator<<(ostream& os, const Matrix& X);
    friend Matrix& operator+=(Matrix& X, const Matrix& Y);
    friend Matrix& operator-=(Matrix& X, const Matrix& Y);
    friend Matrix& operator*=(Matrix& X, double num);
    friend Matrix& operator*=(Matrix& X, const Matrix& Y);
    friend Matrix& operator/=(Matrix& X, double num);
    friend Matrix matcmp(const Matrix& X, const Matrix& Y);
    //  Element-wise multiplication
    friend Matrix emul(const Matrix& X, const Matrix& Y);
    //  Concatenates along the `dimension`:  1 - vertical, 2 - horizontal.
    //  If no value is specified, then the default is the first dimension.
    friend Matrix concat(const Matrix X, Matrix Y, int dimension);
    //  Sums along the `dimension` specified as a positive integer.
    //  If no value is specified, then the default is the first dimension.
    //  sum(A,1) returns a row vector of the sums of each column.
    //  sum(A,2) returns a column vector of the sums of each row.
    friend Matrix sum(const Matrix& X, int dimension);
    //  Returns two matrices which are two row vectors containing the
    //  maximum element from each column and the indices of those elements.
    //  If the values along the column contain more than one maximal element,
    //  the index of the first one is returned.
    friend tuple<Matrix, Matrix> max(const Matrix &X);
    friend Matrix submat(const Matrix& X, 
                         int startrow, int endrow, int startcol, int endcol);
    friend Matrix submat(const Matrix& X, vector<size_t> indices);
    // Transposition (not cache friendly)
    friend Matrix t(const Matrix& X);
    friend Matrix apply(Matrix X, function<double(double)> func);
};

inline Matrix& operator+=(Matrix& X, const Matrix& Y) {
    assert_msg(X.rows == Y.rows && X.cols == Y.cols, "\n - Error using +"
        "\n - Matrix dimensions must agree.");
    for (size_t i = 0; i < X.size(); i++)
        X.mat[i] += Y.mat[i];
    return X;
}
inline Matrix operator+(const Matrix& X, const Matrix& Y) {
    Matrix ret = X;
    ret += Y;
    return ret;
}
inline Matrix operator*(const Matrix& X, const Matrix& Y) {
    Matrix result = X;
    result *= Y;
    return result;
}
inline Matrix& operator*=(Matrix& X, double num) {
    for (size_t i = 0; i < X.size(); i++)
        X.mat[i] *= num;
    return X;
}
inline Matrix operator*(const Matrix& X, double num) {
    Matrix ret = X;
    ret *= num;
    return ret;
}
inline Matrix operator*(double num, const Matrix& X) {
    return X * num;
}
inline Matrix& operator/=(Matrix& X, double num) {
    for (size_t i = 0; i < X.size(); i++)
        X.mat[i] /= num;
    return X;
}
inline Matrix operator/(const Matrix& X, double num) {
    Matrix ret = X;
    ret /= num;
    return ret;
}
inline Matrix& operator-=(Matrix& X, const Matrix& Y) {
    assert_msg(X.rows == Y.rows && X.cols == Y.cols, "\n - Error using -"
        "\n - Matrix dimensions must agree.");
    for (size_t i = 0; i < X.size(); i++)
        X.mat[i] -= Y.mat[i];
    return X;
}
inline Matrix operator-(const Matrix& X, const Matrix& Y) {
    Matrix ret = X;
    ret -= Y;
    return ret;
}
Matrix concat(Matrix X, Matrix Y, int dimension = 1);
Matrix sum(const Matrix& X, int dimension = 1);
ostream& format_short(ostream& os);
ostream& format_long(ostream& os);

class Zeros : public Matrix {
public:
    Zeros(size_t rows);
    Zeros(size_t rows, size_t cols);
    Zeros(const Zeros&) = delete;
};

class Ones : public Matrix {
public:
    Ones(size_t rows);
    Ones(size_t rows, size_t cols);
    Ones(const Ones&) = delete;
};

class Rand : public Matrix {
public:
    Rand(size_t rows);
    Rand(size_t rows, size_t cols);
    Rand(const Rand&) = delete;
};

class Eye : public Matrix {
public:
    Eye(size_t rows);
    Eye(size_t rows, size_t cols);
    Eye(const Eye&) = delete;
};

} // namespace toymat

#endif