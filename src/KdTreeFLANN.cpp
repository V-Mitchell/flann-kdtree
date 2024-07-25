#include "KdTreeFLANN.hpp"

template <typename T> KdTreeFLANN<T>::KdTreeFLANN(const std::vector<T> &points)
{
    static_assert(std::is_same<Point2D, T>::value || std::is_same<Point3D, T>::value,
                  "T must be either Point2D or Point3D");

    flann::Index<flann::L2<double>> index(pointsToMatrix(points), flann::KDTreeIndexParams(4));
    _index = std::make_unique<flann::Index<flann::L2<double>>>(index);
    _index->buildIndex();
}

template <typename T> KdTreeFLANN<T>::~KdTreeFLANN() { _index.reset(); }

template <typename T> size_t KdTreeFLANN<T>::size() { return _index->size(); }

template <typename T> void KdTreeFLANN<T>::addPoints(const std::vector<T> &points)
{
    _index->addPoints(pointsToMatrix(points));
}

template <typename T> void KdTreeFLANN<T>::removePoints(const std::vector<size_t> &indices)
{
    for (const auto &idx : indices)
    {
        _index->removePoint(idx);
    }
}

template <typename T> std::vector<T> KdTreeFLANN<T>::getPoints()
{
    std::vector<T> points;
    for (size_t i = 0; i < _index->size(); ++i)
    {
        emplaceBackPointData(_index->getPoint(i), points);
    }
    return points;
}

template <typename T> std::vector<T> KdTreeFLANN<T>::getPoints(const std::vector<size_t> &indices)
{
    std::vector<T> points;
    for (const auto &idx : indices)
    {
        emplaceBackPointData(_index->getPoint(idx), points);
    }
    return points;
}

template <typename T>
std::pair<std::vector<std::vector<size_t>>, std::vector<std::vector<double>>>
KdTreeFLANN<T>::knnSearch(const std::vector<T> &query, const size_t &knn)
{
    std::vector<std::vector<size_t>> indices;
    std::vector<std::vector<double>> dists;
    _index->knnSearch(pointsToMatrix(query), indices, dists, knn, flann::SearchParams(128));
    return std::make_pair(indices, dists);
}

template <typename T>
std::pair<std::vector<std::vector<size_t>>, std::vector<std::vector<double>>>
KdTreeFLANN<T>::radiusSearch(const std::vector<T> &query, const float &radius)
{
    std::vector<std::vector<size_t>> indices;
    std::vector<std::vector<double>> dists;
    _index->radiusSearch(pointsToMatrix(query), indices, dists, radius, flann::SearchParams(128));
    return std::make_pair(indices, dists);
}

template <typename T>
flann::Matrix<double> KdTreeFLANN<T>::pointsToMatrix(const std::vector<Point3D> &points)
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

template <typename T>
flann::Matrix<double> KdTreeFLANN<T>::pointsToMatrix(const std::vector<Point2D> &points)
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

template class KdTreeFLANN<Point3D>;
template class KdTreeFLANN<Point2D>;
