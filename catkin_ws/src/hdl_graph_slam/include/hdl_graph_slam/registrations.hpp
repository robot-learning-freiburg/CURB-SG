// SPDX-License-Identifier: BSD-2-Clause

#ifndef HDL_GRAPH_SLAM_REGISTRATIONS_HPP
#define HDL_GRAPH_SLAM_REGISTRATIONS_HPP

#include <ros/ros.h>

#include <pcl/registration/registration.h>
#include <hdl_graph_slam/custom_point_types.hpp>


namespace hdl_graph_slam {

/**
 * @brief select a scan matching algorithm according to rosparams
 * @param pnh
 * @return selected scan matching
 */
    pcl::Registration<pcl::PointXYZINormal, pcl::PointXYZINormal>::Ptr
    select_registration_method(ros::NodeHandle &pnh);

}  // namespace hdl_graph_slam

#endif  //
