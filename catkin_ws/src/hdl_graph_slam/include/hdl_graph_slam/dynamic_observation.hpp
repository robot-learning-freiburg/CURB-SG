// SPDX-License-Identifier: BSD-2-Clause
#include <ros/ros.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <boost/optional.hpp>
#include "geometry_msgs/Pose.h"
#include <hdl_graph_slam/custom_point_types.hpp>
#include <hdl_graph_slam/keyframe.hpp>

namespace g2o {
    class VertexSE3;

    class HyperGraph;

    class SparseOptimizer;
}  // namespace g2o

namespace hdl_graph_slam {

/**
 * @brief Dynamic Observation
 */
    struct DynObservation {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        using PointT = pcl::PointXYZINormal;
        using Ptr = std::shared_ptr<DynObservation>;

        DynObservation(const int id,
                       const ros::Time &stamp,
                       const int keyframe_id,
                       const int vehicle_id,
                       const int object_tag,
                       const geometry_msgs::Pose transform);
//        DynObservation(const int id,
//                       const ros::Time &stamp,
//                       const int prev_id,
//                       const int keyframe_id,
//                       const int vehicle_id,
//                       const Eigen::Isometry3d &transform,
//                       const g2o::VertexSE3 *node) : id(id), stamp(stamp), keyframe_id(keyframe_id),
//                                                     vehicle_id(vehicle_id), transformFromKeyframe(transform);

        virtual ~DynObservation();

    public:
        unsigned int id;
        ros::Time stamp;
        unsigned int prev_id;
        int keyframe_id;
        int vehicle_id;
        int object_tag;
        int agent_no;
        geometry_msgs::Pose transformFromKeyframe;
        pcl::PointCloud<PointT>::ConstPtr cloud;
        hdl_graph_slam::KeyFrame::Ptr keyframe;
        g2o::VertexSE3 *node;  // node instance
    };

}  // namespace hdl_graph_slam
