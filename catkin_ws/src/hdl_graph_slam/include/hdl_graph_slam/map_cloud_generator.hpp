// SPDX-License-Identifier: BSD-2-Clause

#ifndef MAP_CLOUD_GENERATOR_HPP
#define MAP_CLOUD_GENERATOR_HPP

#include <vector>
#include <pcl/point_types.h>
#include <hdl_graph_slam/custom_point_types.hpp>

#include <pcl/point_cloud.h>
#include <hdl_graph_slam/keyframe.hpp>

namespace hdl_graph_slam {

/**
 * @brief this class generates a map point cloud from registered keyframes
 */
    class MapCloudGenerator {
    public:
        using PointT = pcl::PointXYZINormal;

        MapCloudGenerator();

        ~MapCloudGenerator();

        /**
         * @brief generates a map point cloud
         * @param keyframes   snapshots of keyframes
         * @param resolution  resolution of generated map
         * @return generated map point cloud
         */
        pcl::PointCloud<PointT>::Ptr
        generate(const std::vector <KeyFrameSnapshot::Ptr> &keyframes, double resolution_rough,
                 double resolution_medium, double resulution_fine, std::vector<int> reduce_classes) const;

        pcl::PointCloud<PointT>::Ptr
        filterPointCloud(
                const pcl::PointCloud<PointT>::Ptr &input_cloud,
                const std::vector<int> classes_for_higher_resolution,
                const float resolution) const;

        std::vector<int> classes_for_high_resolution = {5, 12, 18};
        std::vector<int> classes_for_medium_resolution = {17, 19, 20};
    };

}  // namespace hdl_graph_slam

#endif  // MAP_POINTCLOUD_GENERATOR_HPP
