#include "KdTreeFLANN.hpp"

int main(int argc, char **argv)
{
    std::cout << "// SIMPLE EXAMPLE //" << std::endl;
    std::cout << "// 3D Points //" << std::endl;
    {
        const int nn = 2;
        // matrix is row major order
        flann::Matrix<double> dataset(
            new double[12]{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0}, 4, 3);
        flann::Matrix<double> query(new double[6]{1.0, 2.0, 3.0, 10.0, 11.0, 12.0}, 2, 3);

        flann::Matrix<int> indices(new int[query.rows * nn], query.rows, nn);
        flann::Matrix<double> dists(new double[query.rows * nn], query.rows, nn);

        flann::Index<flann::L2<double>> index(dataset, flann::KDTreeIndexParams(4));
        index.buildIndex();

        index.knnSearch(query, indices, dists, nn, flann::SearchParams(128));

        printMatrix<int>(indices);
        printMatrix<double>(dists);

        delete[] dataset.ptr();
        delete[] query.ptr();
        delete[] indices.ptr();
        delete[] dists.ptr();
    }
    std::cout << "// 2D Points //" << std::endl;
    {
        const int nn = 2;
        // matrix is row major order
        flann::Matrix<double> dataset(new double[12]{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}, 4, 2);
        flann::Matrix<double> query(new double[6]{1.0, 2.0, 7.0, 8.0}, 2, 2);

        flann::Matrix<int> indices(new int[query.rows * nn], query.rows, nn);
        flann::Matrix<double> dists(new double[query.rows * nn], query.rows, nn);

        flann::Index<flann::L2<double>> index(dataset, flann::KDTreeIndexParams(4));
        index.buildIndex();

        index.knnSearch(query, indices, dists, nn, flann::SearchParams(128));

        printMatrix<int>(indices);
        printMatrix<double>(dists);

        delete[] dataset.ptr();
        delete[] query.ptr();
        delete[] indices.ptr();
        delete[] dists.ptr();
    }

    std::cout << "// LIBRARY TEST //" << std::endl;
    std::cout << "// 3D Points //" << std::endl;
    {
        const int nn = 2;
        std::vector<Point3D> points = {Point3D(1.0, 2.0, 3.0), Point3D(4.0, 5.0, 6.0),
                                       Point3D(7.0, 8.0, 9.0), Point3D(10.0, 11.0, 12.0)};
        std::vector<Point3D> query = {Point3D(1.0, 2.0, 3.0), Point3D(10.0, 11.0, 12.0)};
        KdTreeFLANN<Point3D> kdTree(points);
        const auto searchResult = kdTree.knnSearch(query, nn);
        const std::vector<std::vector<size_t>> &indices = searchResult.first;
        const std::vector<std::vector<double>> &dists = searchResult.second;
        for (const auto &vec : indices)
        {
            for (const auto &idx : vec)
            {
                std::cout << idx << " ";
            }
            std::cout << std::endl;
        }
        for (const auto &vec : dists)
        {
            for (const auto &dist : vec)
            {
                std::cout << dist << " ";
            }
            std::cout << std::endl;
        }
    }
    std::cout << "// 2D Points //" << std::endl;
    {
        const int nn = 2;
        std::vector<Point2D> points = {Point2D(1.0, 2.0), Point2D(3.0, 4.0), Point2D(5.0, 6.0),
                                       Point2D(7.0, 8.0)};
        std::vector<Point2D> query = {Point2D(1.0, 2.0), Point2D(7.0, 8.0)};
        KdTreeFLANN<Point2D> kdTree(points);
        const auto searchResult = kdTree.knnSearch(query, nn);
        const std::vector<std::vector<size_t>> &indices = searchResult.first;
        const std::vector<std::vector<double>> &dists = searchResult.second;
        for (const auto &vec : indices)
        {
            for (const auto &idx : vec)
            {
                std::cout << idx << " ";
            }
            std::cout << std::endl;
        }
        for (const auto &vec : dists)
        {
            for (const auto &dist : vec)
            {
                std::cout << dist << " ";
            }
            std::cout << std::endl;
        }
    }

    std::cout << "Finished" << std::endl;

    return 0;
}
