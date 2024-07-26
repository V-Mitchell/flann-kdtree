#include "../src/KdTreeFLANN.hpp"
#include <gtest/gtest.h>

namespace
{

TEST(KdTreeFLANNTest, Build2DKdTree)
{
    std::vector<Point2D> points = {Point2D(1.0, 2.0), Point2D(3.0, 4.0), Point2D(5.0, 6.0),
                                   Point2D(7.0, 8.0)};
    EXPECT_EQ(points.size(), 4);
    KdTreeFLANN<Point2D> kdTree(points);

    ASSERT_EQ(points.size(), kdTree.size())
        << "Size of 2D points vector {" << points.size() << "} and KdTree points {" << kdTree.size()
        << "} is unequal";
}

TEST(KdTreeFLANNTest, Build3DKdTree)
{
    std::vector<Point3D> points = {Point3D(1.0, 2.0, 3.0), Point3D(4.0, 5.0, 6.0),
                                   Point3D(7.0, 8.0, 9.0), Point3D(10.0, 11.0, 12.0)};
    EXPECT_EQ(points.size(), 4);
    KdTreeFLANN<Point3D> kdTree(points);

    ASSERT_EQ(points.size(), kdTree.size())
        << "Size of 3D points vector {" << points.size() << "} and KdTree points {" << kdTree.size()
        << "} is unequal";
}

TEST(KdTreeFLANNTest, KNNSearch2D)
{
    std::vector<std::vector<size_t>> indicesGT = {{0, 1}, {3, 2}};
    std::vector<std::vector<size_t>> distsGT = {{0, 8}, {0, 8}};

    std::vector<Point2D> points = {Point2D(1.0, 2.0), Point2D(3.0, 4.0), Point2D(5.0, 6.0),
                                   Point2D(7.0, 8.0)};
    KdTreeFLANN<Point2D> kdTree(points);

    const size_t knn = 2;
    std::vector<Point2D> query = {Point2D(1.0, 2.0), Point2D(7.0, 8.0)};
    const auto searchResult = kdTree.knnSearch(query, knn);
    const std::vector<std::vector<size_t>> &indices = searchResult.first;
    const std::vector<std::vector<double>> &dists = searchResult.second;

    EXPECT_EQ(indices.size(), query.size())
        << "KNN Search result indices size {" << indices.size() << "} and number of queries {"
        << query.size() << "} is unequal";
    EXPECT_EQ(dists.size(), query.size())
        << "KNN Search result dists size {" << indices.size() << "} and number of queries {"
        << query.size() << "} is unequal";

    for (size_t i = 0; i < indices.size(); ++i)
    {
        EXPECT_EQ(indices[i].size(), knn)
            << "KNN Search result query indices size {" << indices[i].size()
            << "} and number of nearest neighbors {" << knn << "} is unequal";
        for (size_t j = 0; j < indices[i].size(); ++j)
        {
            EXPECT_EQ(indices[i][j], indicesGT[i][j])
                << "KNN Search result query index {" << indices[i][j] << "} is less than 0";
        }
    }

    for (size_t i = 0; i < dists.size(); ++i)
    {
        EXPECT_EQ(dists[i].size(), knn)
            << "KNN Search result query dists size {" << dists[i].size()
            << "} and number of nearest neighbors {" << knn << "} is unequal";
        for (size_t j = 0; j < dists[i].size(); ++j)
        {
            EXPECT_EQ(dists[i][j], distsGT[i][j])
                << "KNN Search result query dist {" << dists[i][j] << "} is less than 0";
        }
    }
}

TEST(KdTreeFLANNTest, KNNSearch3D)
{
    std::vector<std::vector<size_t>> indicesGT = {{0, 1}, {3, 2}};
    std::vector<std::vector<size_t>> distsGT = {{0, 27}, {0, 27}};

    std::vector<Point3D> points = {Point3D(1.0, 2.0, 3.0), Point3D(4.0, 5.0, 6.0),
                                   Point3D(7.0, 8.0, 9.0), Point3D(10.0, 11.0, 12.0)};
    KdTreeFLANN<Point3D> kdTree(points);

    const size_t knn = 2;
    std::vector<Point3D> query = {Point3D(1.0, 2.0, 3.0), Point3D(10.0, 11.0, 12.0)};
    const auto searchResult = kdTree.knnSearch(query, knn);
    const std::vector<std::vector<size_t>> &indices = searchResult.first;
    const std::vector<std::vector<double>> &dists = searchResult.second;

    EXPECT_EQ(indices.size(), query.size())
        << "KNN Search result indices size {" << indices.size() << "} and number of queries {"
        << query.size() << "} is unequal";
    EXPECT_EQ(dists.size(), query.size())
        << "KNN Search result dists size {" << indices.size() << "} and number of queries {"
        << query.size() << "} is unequal";

    for (size_t i = 0; i < indices.size(); ++i)
    {
        EXPECT_EQ(indices[i].size(), knn)
            << "KNN Search result query indices size {" << indices[i].size()
            << "} and number of nearest neighbors {" << knn << "} is unequal";
        for (size_t j = 0; j < indices[i].size(); ++j)
        {
            EXPECT_EQ(indices[i][j], indicesGT[i][j])
                << "KNN Search result query index {" << indices[i][j] << "} is less than 0";
        }
    }

    for (size_t i = 0; i < dists.size(); ++i)
    {
        EXPECT_EQ(dists[i].size(), knn)
            << "KNN Search result query dists size {" << dists[i].size()
            << "} and number of nearest neighbors {" << knn << "} is unequal";
        for (size_t j = 0; j < dists[i].size(); ++j)
        {
            EXPECT_EQ(dists[i][j], distsGT[i][j])
                << "KNN Search result query dist {" << dists[i][j] << "} is less than 0";
        }
    }
}

TEST(KdTreeFLANNTest, RadiusSearch2D)
{
    const std::vector<std::vector<size_t>> indicesGT = {{0, 1, 2}, {3, 2, 1}};
    const std::vector<std::vector<size_t>> distsGT = {{0, 8, 32}, {0, 8, 32}};
    const size_t numPtsGT = 3;

    const std::vector<Point2D> points = {Point2D(1.0, 2.0), Point2D(3.0, 4.0), Point2D(5.0, 6.0),
                                         Point2D(7.0, 8.0)};
    KdTreeFLANN<Point2D> kdTree(points);

    const float radius = 50.0;
    const std::vector<Point2D> query = {Point2D(1.0, 2.0), Point2D(7.0, 8.0)};
    const auto searchResult = kdTree.radiusSearch(query, radius);
    const std::vector<std::vector<size_t>> &indices = searchResult.first;
    const std::vector<std::vector<double>> &dists = searchResult.second;

    EXPECT_EQ(indices.size(), query.size())
        << "Radius Search result indices size {" << indices.size() << "} and number of queries {"
        << query.size() << "} is unequal";
    EXPECT_EQ(dists.size(), query.size())
        << "Radius Search result dists size {" << indices.size() << "} and number of queries {"
        << query.size() << "} is unequal";

    for (size_t i = 0; i < indices.size(); ++i)
    {
        EXPECT_EQ(indices[i].size(), numPtsGT)
            << "Radius Search result query indices size {" << indices[i].size()
            << "} and number of radius neighbors {" << numPtsGT << "} is unequal";
        for (size_t j = 0; j < indices[i].size(); ++j)
        {
            EXPECT_EQ(indices[i][j], indicesGT[i][j])
                << "Radius Search result query index {" << indices[i][j] << "} is less than 0";
        }
    }

    for (size_t i = 0; i < dists.size(); ++i)
    {
        EXPECT_EQ(dists[i].size(), numPtsGT)
            << "Radius Search result query dists size {" << dists[i].size()
            << "} and number of radius neighbors {" << numPtsGT << "} is unequal";
        for (size_t j = 0; j < dists[i].size(); ++j)
        {
            EXPECT_EQ(dists[i][j], distsGT[i][j])
                << "Radius Search result query dist {" << dists[i][j] << "} is less than 0";
        }
    }
}

TEST(KdTreeFLANNTest, RadiusSearch3D)
{
    const std::vector<std::vector<size_t>> indicesGT = {{0, 1, 2}, {3, 2, 1}};
    const std::vector<std::vector<size_t>> distsGT = {{0, 27, 108}, {0, 27, 108}};
    const size_t numPtsGT = 3;

    const std::vector<Point3D> points = {Point3D(1.0, 2.0, 3.0), Point3D(4.0, 5.0, 6.0),
                                         Point3D(7.0, 8.0, 9.0), Point3D(10.0, 11.0, 12.0)};
    KdTreeFLANN<Point3D> kdTree(points);

    const float radius = 200.0;
    const std::vector<Point3D> query = {Point3D(1.0, 2.0, 3.0), Point3D(10.0, 11.0, 12.0)};
    const auto searchResult = kdTree.radiusSearch(query, radius);
    const std::vector<std::vector<size_t>> &indices = searchResult.first;
    const std::vector<std::vector<double>> &dists = searchResult.second;

    EXPECT_EQ(indices.size(), query.size())
        << "Radius Search result indices size {" << indices.size() << "} and number of queries {"
        << query.size() << "} is unequal";
    EXPECT_EQ(dists.size(), query.size())
        << "Radius Search result dists size {" << indices.size() << "} and number of queries {"
        << query.size() << "} is unequal";

    for (size_t i = 0; i < indices.size(); ++i)
    {
        EXPECT_EQ(indices[i].size(), numPtsGT)
            << "Radius Search result query indices size {" << indices[i].size()
            << "} and number of radius neighbors {" << numPtsGT << "} is unequal";
        for (size_t j = 0; j < indices[i].size(); ++j)
        {
            EXPECT_EQ(indices[i][j], indicesGT[i][j])
                << "Radius Search result query index {" << indices[i][j] << "} is less than 0";
        }
    }

    for (size_t i = 0; i < dists.size(); ++i)
    {
        EXPECT_EQ(dists[i].size(), numPtsGT)
            << "Radius Search result query dists size {" << dists[i].size()
            << "} and number of radius neighbors {" << numPtsGT << "} is unequal";
        for (size_t j = 0; j < dists[i].size(); ++j)
        {
            EXPECT_EQ(dists[i][j], distsGT[i][j])
                << "Radius Search result query dist {" << dists[i][j] << "} is less than 0";
        }
    }
}

} // namespace

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}