// SPDX-License-Identifier: BSD-2-Clause

#include <hdl_graph_slam/map_cloud_generator.hpp>
#include <algorithm>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/octree/octree_search.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/conditional_removal.h>

namespace hdl_graph_slam {

    MapCloudGenerator::MapCloudGenerator() {}

    MapCloudGenerator::~MapCloudGenerator() {}

    pcl::PointCloud<MapCloudGenerator::PointT>::Ptr
    MapCloudGenerator::filterPointCloud(
            const pcl::PointCloud<PointT>::Ptr &input_cloud,
            const std::vector<int> classes_for_higher_resolution,
            const float resolution) const {
        // Apply filter to point cloud
        pcl::PointCloud<PointT>::Ptr cloud_of_small_objects(new pcl::PointCloud <PointT>);
        pcl::ConditionOr<PointT>::Ptr range_cond(new pcl::ConditionOr<PointT>());

        for (int i = 0; i < classes_for_higher_resolution.size(); i++) {
            range_cond->addComparison(pcl::FieldComparison<PointT>::ConstPtr(
                    new pcl::FieldComparison<PointT>("intensity", pcl::ComparisonOps::EQ,
                                                     classes_for_higher_resolution[i])));
        }
        // build the filter
        pcl::ConditionalRemoval <PointT> condrem;
        condrem.setCondition(range_cond);
        condrem.setInputCloud(input_cloud);
        condrem.setKeepOrganized(true);

        // apply filter
        condrem.filter(*cloud_of_small_objects);

        // special filter for the small objects
        std::cout << "filter for small objects" << std::endl;
        pcl::Filter<PointT>::Ptr downsample_filter_for_small_objects;
        auto voxelgrid_for_small_objects = new pcl::VoxelGrid<PointT>();
        voxelgrid_for_small_objects->setLeafSize(resolution, resolution, resolution);
        downsample_filter_for_small_objects.reset(voxelgrid_for_small_objects);
        downsample_filter_for_small_objects->setInputCloud(cloud_of_small_objects);
        pcl::PointCloud<PointT>::Ptr filtered_cloud_of_small_objects(new pcl::PointCloud<PointT>());
        downsample_filter_for_small_objects->filter(*filtered_cloud_of_small_objects);

        return filtered_cloud_of_small_objects;
    }

    pcl::PointCloud<MapCloudGenerator::PointT>::Ptr
    MapCloudGenerator::generate(const std::vector <KeyFrameSnapshot::Ptr> &keyframes, double resolution_rough,
                                double resolution_medium, double resolution_fine,
                                std::vector<int> reduce_classes) const {
        if (keyframes.empty()) {
            std::cerr << "warning: keyframes empty!!" << std::endl;
            return nullptr;
        }

        pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
        std::cout << "before reserve: " << keyframes.front()->cloud->size() << " * " << keyframes.size() << " = "
                  << keyframes.front()->cloud->size() * keyframes.size() << std::endl;
        cloud->reserve(keyframes.front()->cloud->size() * keyframes.size());

        for (const auto &keyframe: keyframes) {
            Eigen::Matrix4f pose = keyframe->pose.matrix().cast<float>();
            bool reduce = false;
            for (const auto &src_pt: keyframe->cloud->points) {

                PointT dst_pt;
                dst_pt.getVector4fMap() = pose * src_pt.getVector4fMap();
                dst_pt.intensity = src_pt.intensity;
                dst_pt.curvature = src_pt.curvature;

                cloud->push_back(dst_pt);
            }
        }

        pcl::PointCloud<PointT>::Ptr filtered_cloud_high_res = filterPointCloud(cloud, classes_for_high_resolution,
                                                                                resolution_fine);
        pcl::PointCloud<PointT>::Ptr filtered_cloud_medium_res = filterPointCloud(cloud, classes_for_medium_resolution,
                                                                                  resolution_medium);


        cloud->width = cloud->size();
        cloud->height = 1;
        cloud->is_dense = false;

        if (resolution_rough <= 0.0)
            return cloud; // To get unfiltered point cloud with intensity


        // filter for the whole cloud
        std::cout << "Filter for the whole cloud" << std::endl;
        pcl::Filter<PointT>::Ptr downsample_filter;
        auto voxelgrid = new pcl::VoxelGrid<PointT>();
        voxelgrid->setLeafSize(resolution_rough, resolution_rough, resolution_rough);
        downsample_filter.reset(voxelgrid);

        downsample_filter->setInputCloud(cloud);
        pcl::PointCloud<PointT>::Ptr filtered_cloud(new pcl::PointCloud<PointT>());
        downsample_filter->filter(*filtered_cloud);

        *filtered_cloud += *filtered_cloud_high_res;
        *filtered_cloud += *filtered_cloud_medium_res;

        return filtered_cloud;
    }

}  // namespace hdl_graph_slam
