#include "KdTreeFLANN.hpp"

flann::Matrix<double> pointsToMatrix(const std::vector<Point3D> &points)
{
    const size_t dims = 3;
    const size_t &size = points.size();
    double *data = new double[points.size() * dims]; // Memory Leak?
    for (size_t i = 0; i < points.size(); ++i)
    {
        data[i * dims] = points[i].x;
        data[i * dims + 1] = points[i].y;
        data[i * dims + 2] = points[i].z;
    }
    return flann::Matrix<double>(data, size, dims);
}

flann::Matrix<double> pointsToMatrix(const std::vector<Point2D> &points)
{
    const size_t dims = 2;
    const size_t &size = points.size();
    double *data = new double[points.size() * dims]; // Memory Leak?
    for (size_t i = 0; i < points.size(); ++i)
    {
        data[i * dims] = points[i].x;
        data[i * dims + 1] = points[i].y;
    }
    return flann::Matrix<double>(data, size, dims);
}

KdTreeFLANN::KdTreeFLANN(const std::vector<Point3D> &points)
{
    flann::Index<flann::L2<double>> index(pointsToMatrix(points), flann::KDTreeIndexParams(4));
    _index = std::make_unique<flann::Index<flann::L2<double>>>(index);
    _index->buildIndex();
}

KdTreeFLANN::~KdTreeFLANN() { _index.reset(); }

std::pair<std::vector<std::vector<size_t>>, std::vector<std::vector<double>>>
KdTreeFLANN::knnSearch(const std::vector<Point3D> &query, const size_t &knn)
{

    std::vector<std::vector<size_t>> indices;
    std::vector<std::vector<double>> dists;
    _index->knnSearch(pointsToMatrix(query), indices, dists, knn, flann::SearchParams(128));
    return std::make_pair(indices, dists);
}

std::pair<std::vector<std::vector<size_t>>, std::vector<std::vector<double>>>
KdTreeFLANN::radiusSearch(const std::vector<Point3D> &query, const float &radius)
{
    std::vector<std::vector<size_t>> indices;
    std::vector<std::vector<double>> dists;
    _index->radiusSearch(pointsToMatrix(query), indices, dists, radius, flann::SearchParams(128));
    return std::make_pair(indices, dists);
}

template <typename T> KdTreeTest<T>::KdTreeTest(const std::vector<T> &points)
{
    static_assert(std::is_same<Point2D, T>::value || std::is_same<Point3D, T>::value,
                  "T must be either Point2D or Point3D");

    flann::Index<flann::L2<double>> index(pointsToMatrix(points), flann::KDTreeIndexParams(4));
    _index = std::make_unique<flann::Index<flann::L2<double>>>(index);
    _index->buildIndex();
}

template <typename T> KdTreeTest<T>::~KdTreeTest() { _index.reset(); }

template <typename T>
std::pair<std::vector<std::vector<size_t>>, std::vector<std::vector<double>>>
KdTreeTest<T>::knnSearch(const std::vector<T> &query, const size_t &knn)
{
    std::vector<std::vector<size_t>> indices;
    std::vector<std::vector<double>> dists;
    _index->knnSearch(pointsToMatrix(query), indices, dists, knn, flann::SearchParams(128));
    return std::make_pair(indices, dists);
}

template <typename T>
std::pair<std::vector<std::vector<size_t>>, std::vector<std::vector<double>>>
KdTreeTest<T>::radiusSearch(const std::vector<T> &query, const float &radius)
{
    std::vector<std::vector<size_t>> indices;
    std::vector<std::vector<double>> dists;
    _index->radiusSearch(pointsToMatrix(query), indices, dists, radius, flann::SearchParams(128));
    return std::make_pair(indices, dists);
}
