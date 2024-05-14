// SPDX-License-Identifier: BSD-2-Clause

#include <hdl_graph_slam/dynamic_observation.hpp>

#include <boost/filesystem.hpp>
#include "geometry_msgs/Pose.h"
#include <pcl/io/pcd_io.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/types/slam3d/vertex_se3.h>

namespace hdl_graph_slam {

    DynObservation::DynObservation(const int id,
                                   const ros::Time &stamp,
                                   const int keyframe_id,
                                   const int vehicle_id,
                                   const int object_tag,
                                   const geometry_msgs::Pose transform) : id(id), stamp(stamp),
                                                                          keyframe_id(keyframe_id),
                                                                          vehicle_id(vehicle_id),
                                                                          object_tag(object_tag),
                                                                          transformFromKeyframe(transform) {}

    DynObservation::~DynObservation() {}

}  // namespace hdl_graph_slam
