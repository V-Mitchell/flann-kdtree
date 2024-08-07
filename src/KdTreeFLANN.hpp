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
    struct FLANNConfig
    {
        FLANNConfig()
            : numTrees(4)
            , numSearchChecks(128)
        {
        }

        FLANNConfig(const int &numTrees, const int &numSearchChecks)
            : numTrees(numTrees)
            , numSearchChecks(numSearchChecks)
        {
        }

        int numTrees;
        int numSearchChecks;
    };

    KdTreeFLANN(const std::vector<T> &points, const FLANNConfig &config = FLANNConfig());
    ~KdTreeFLANN();

    void addPoints(const std::vector<T> &points);

    void removePoints(const std::vector<size_t> &indices);

    std::vector<T> getPoints();

    std::vector<T> getPoints(const std::vector<size_t> &indices);

    size_t size();

    std::pair<std::vector<std::vector<size_t>>, std::vector<std::vector<double>>>
    knnSearch(const std::vector<T> &query, const size_t &knn);

    std::pair<std::vector<std::vector<size_t>>, std::vector<std::vector<double>>>
    radiusSearch(const std::vector<T> &query, const float &radius);

  private:
    flann::Matrix<double> pointsToMatrix(const std::vector<Point3D> &points);
    flann::Matrix<double> pointsToMatrix(const std::vector<Point2D> &points);

    inline void emplaceBackPointData(double *dataPtr, std::vector<Point3D> &points)
    {
        points.emplace_back(*dataPtr, *(dataPtr + 1), *(dataPtr + 2));
    }
    inline void emplaceBackPointData(double *dataPtr, std::vector<Point2D> &points)
    {
        points.emplace_back(*dataPtr, *(dataPtr + 1));
    }

    FLANNConfig _config;
    std::unique_ptr<flann::Index<flann::L2<double>>> _index;
};

#endif