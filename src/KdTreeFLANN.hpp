#ifndef KDTREE_FLANN_H_
#define KDTREE_FLANN_H_

#include <flann/flann.hpp>
#include <memory>
#include <type_traits>

struct Point3D
{
    Point3D()
        : x(0)
        , y(0)
        , z(0)
    {
    }

    Point3D(const double &x, const double &y, const double &z)
        : x(x)
        , y(y)
        , z(z)
    {
    }

    double x, y, z;
};

struct Point2D
{
    Point2D()
        : x(0)
        , y(0)
    {
    }

    Point2D(const double &x, const double &y)
        : x(x)
        , y(y)
    {
    }

    double x, y;
};

template <typename T> void printMatrix(const flann::Matrix<T> &matrix)
{
    const size_t &rows = matrix.rows;
    const size_t &cols = matrix.cols;
    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
        {
            std::cout << *(matrix.ptr() + i * cols + j) << " ";
        }
        std::cout << std::endl;
    }
}

template <typename T> class KdTreeFLANN
{
  public:
    KdTreeFLANN(const std::vector<T> &points);
    ~KdTreeFLANN();

    size_t size();

    std::pair<std::vector<std::vector<size_t>>, std::vector<std::vector<double>>>
    knnSearch(const std::vector<T> &query, const size_t &knn);

    std::pair<std::vector<std::vector<size_t>>, std::vector<std::vector<double>>>
    radiusSearch(const std::vector<T> &query, const float &radius);

  private:
    flann::Matrix<double> pointsToMatrix(const std::vector<Point3D> &points);
    flann::Matrix<double> pointsToMatrix(const std::vector<Point2D> &points);

    std::unique_ptr<flann::Index<flann::L2<double>>> _index;
};

#endif