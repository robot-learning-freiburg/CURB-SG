// SPDX-License-Identifier: BSD-2-Clause

#include <ctime>
#include <mutex>
#include <atomic>
#include <memory>
#include <iomanip>
#include <iostream>
#include <unordered_map>
#include <boost/format.hpp>
#include <boost/thread.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>

#include <Eigen/Dense>
#include <eigen_conversions/eigen_msg.h>

#include <ros/ros.h>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>
#include "pcl_ros/transforms.h"
#include "pcl_ros/impl/transforms.hpp"
#include <pcl/filters/random_sample.h>
#include <tf_conversions/tf_eigen.h>
#include <tf/transform_listener.h>
#include <std_msgs/Time.h>
#include <std_msgs/String.h>
#include <std_msgs/Header.h>
#include <geodesy/utm.h>
#include <geodesy/wgs84.h>
#include <geographic_msgs/GeoPointStamped.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/NavSatFix.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>
#include <nmea_msgs/Sentence.h>
#include <visualization_msgs/MarkerArray.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <jsk_recognition_msgs/BoundingBox.h>
#include <jsk_recognition_msgs/BoundingBoxArray.h>

#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>

#include <hdl_graph_slam/FloorCoeffs.h>
#include <hdl_graph_slam/SaveMap.h>
#include <hdl_graph_slam/DumpGraph.h>
#include <hdl_graph_slam/ros_utils.hpp>
#include <hdl_graph_slam/ros_time_hash.hpp>
#include <hdl_graph_slam/custom_point_types.hpp>
#include <hdl_graph_slam/graph_slam.hpp>
#include <hdl_graph_slam/keyframe.hpp>
#include <hdl_graph_slam/dynamic_observation.hpp>
#include <hdl_graph_slam/Keyframe_msg.h>
#include <hdl_graph_slam/DynObservation_msg.h>
#include <hdl_graph_slam/ObservationFilter_msg.h>
#include <hdl_graph_slam/DynObservationArray_msg.h>
#include <hdl_graph_slam/keyframe_updater.hpp>
#include <hdl_graph_slam/loop_detector.hpp>
#include <hdl_graph_slam/information_matrix_calculator.hpp>
#include <hdl_graph_slam/map_cloud_generator.hpp>
#include <hdl_graph_slam/nmea_sentence_parser.hpp>

#include <g2o/types/slam3d/edge_se3.h>
#include <g2o/types/slam3d/vertex_se3.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/edge_se3_plane.hpp>
#include <g2o/edge_se3_priorxy.hpp>
#include <g2o/edge_se3_priorxyz.hpp>
#include <g2o/edge_se3_priorvec.hpp>
#include <g2o/edge_se3_priorquat.hpp>

template<typename T>
std::string type_name();

namespace hdl_graph_slam {

    class DynamicObject {
    public:

        explicit DynamicObject(int x) {
            id = x;
        }

        using Ptr = std::shared_ptr<DynamicObject>;
        int id;
        std::deque <KeyFrame::Ptr> dyn_keyframe_queue;
    };

    class MergingEdge {
    public:

        explicit MergingEdge(ros::Time s, int fk, int fn, int tk, int tn, Eigen::Isometry3d rp)
                : stamp(s), from_node_id(fn), to_node_id(tn), from_keyframe_id(fk), to_keyframe_id(tk),
                  relative_pose(rp) {}

        using Ptr = std::shared_ptr<MergingEdge>;
        ros::Time stamp;
        int from_node_id;
        int to_node_id;
        int from_keyframe_id;
        int to_keyframe_id;
        Eigen::Isometry3d relative_pose;

        friend std::ostream &operator<<(std::ostream &os, const MergingEdge &edge) {
            os << "MergingEdge: Keyframe: " <<
               edge.from_keyframe_id << " (Node id: " << edge.from_node_id << ") -> Keyframe: " <<
               edge.to_keyframe_id << " (Node id: " << edge.to_node_id << ")";
            return os;
        }
    };

    class MapServerNodelet : public nodelet::Nodelet {
    public:
        typedef pcl::PointXYZINormal PointT;
        typedef message_filters::sync_policies::ApproximateTime <nav_msgs::Odometry, sensor_msgs::PointCloud2> ApproxSyncPolicy;

        MapServerNodelet() {}


        virtual ~MapServerNodelet() {}

        virtual void onInit() {
            ms_nh = getNodeHandle();
            ms_mt_nh = getMTNodeHandle();
            ms_private_nh = getPrivateNodeHandle();

            // init parameters
            agent_no = ms_private_nh.param<std::string>("agent_no", "101");
            map_server_topic = ms_private_nh.param<std::string>("map_server_topic", "/map_server");
            map_frame_id = ms_private_nh.param<std::string>("map_frame_id", "/map_s");

            map_cloud_resolution_rough = ms_private_nh.param<double>("map_cloud_resolution_rough", 0.6);
            map_cloud_resolution_medium = ms_private_nh.param<double>("map_cloud_resolution_medium", 0.3);
            map_cloud_resolution_fine = ms_private_nh.param<double>("map_cloud_resolution_fine", 0.1);

            agent_keyframe_topic = ms_private_nh.param<std::string>("agent_keyframe_topic", "/agent_keyframes");
            observation_filter_topic = ms_private_nh.param<std::string>("observation_filter_topic",
                                                                        "/observation_filter");
            dynamic_observation_topic = ms_private_nh.param<std::string>("dynamic_observation_topic",
                                                                         "/dynamic_observation");

            height_offset_keyframes = ms_private_nh.param<double>("height_offset_keyframes", 10.0);
            exit_when_map_explored = ms_private_nh.param<bool>("exit_when_map_explored", false);
            exit_iterations = ms_private_nh.param<int>("exit_iterations", 5);
            number_of_map_points_when_expored = ms_private_nh.param<double>("exit_number_of_map_points", 500000.0);

            std::cout << "Exiting when map is explored: " << exit_when_map_explored << std::endl;
            std::cout << "Map is explored with " << number_of_map_points_when_expored << " points." << std::endl;
            std::cout << "Exit iterations: " << exit_iterations << std::endl;

            anchor_node = nullptr;
            anchor_edge = nullptr;
            floor_plane_node = nullptr;
            graph_slam.reset(new GraphSLAM(ms_private_nh.param<std::string>("g2o_solver_type", "lm_var")));
            keyframe_updater.reset(new hdl_graph_slam::KeyframeUpdater(ms_private_nh));
            loop_detector.reset(new hdl_graph_slam::LoopDetector(ms_private_nh));
            map_cloud_generator.reset(new hdl_graph_slam::MapCloudGenerator());
            inf_calclator.reset(new hdl_graph_slam::InformationMatrixCalculator(ms_private_nh));

            // subscribers
            ms_command_sub = ms_nh.subscribe(map_server_topic + "/command", 1024,
                                             &hdl_graph_slam::MapServerNodelet::ms_command_callback, this);

            keyframe_msg_sub = ms_nh.subscribe(agent_keyframe_topic, 1024,
                                               &hdl_graph_slam::MapServerNodelet::keyframe_msg_callback, this);

            dynamic_observation_msg_sub = ms_nh.subscribe(dynamic_observation_topic, 1024,
                                                          &hdl_graph_slam::MapServerNodelet::dynamic_object_msg_callback,
                                                          this);

            observation_filter_sub = ms_nh.subscribe(observation_filter_topic, 1024,
                                                     &hdl_graph_slam::MapServerNodelet::observation_filter_callback,
                                                     this);

            // publishers
            markers_pub = ms_mt_nh.advertise<visualization_msgs::MarkerArray>(map_server_topic + "/markers", 16);
            observation_array_pub = ms_mt_nh.advertise<hdl_graph_slam::DynObservationArray_msg>(
                    map_server_topic + "/observation_array", 4);

            keyframe_metric_pub = ms_mt_nh.advertise<hdl_graph_slam::DynObservationArray_msg>(
                    "/keyframes_for_metric", 4);
            merge_metric_pub = ms_mt_nh.advertise<geometry_msgs::PointStamped>(
                    "/merge_metric", 4);
            map_points_pub = ms_mt_nh.advertise<sensor_msgs::PointCloud2>(map_server_topic + "/map_points", 1, true);
            map_frame_id = "world";

            merge_test_c1 = ms_mt_nh.advertise<sensor_msgs::PointCloud2>(map_server_topic + "/c1", 1, true);
            merge_test_c2 = ms_mt_nh.advertise<sensor_msgs::PointCloud2>(map_server_topic + "/c2", 1, true);
            merge_test_cT = ms_mt_nh.advertise<sensor_msgs::PointCloud2>(map_server_topic + "/cT", 1, true);
            merge_test_final = ms_mt_nh.advertise<sensor_msgs::PointCloud2>(map_server_topic + "/final", 1, true);
            merge_test_kf_old = ms_mt_nh.advertise<sensor_msgs::PointCloud2>(map_server_topic + "/old", 1, true);
            merge_test_kf_new = ms_mt_nh.advertise<sensor_msgs::PointCloud2>(map_server_topic + "/new", 1, true);

            new_run_pub = ms_mt_nh.advertise<std_msgs::Header>("/new_run", 1, true);
            new_run_published = false;

            end_run_pub = ms_mt_nh.advertise<std_msgs::Header>("/end_run", 1, true);
            map_explored_pub = ms_mt_nh.advertise<std_msgs::Header>("/map_explored", 1, true);
            end_run_counter = 0;

            graph_updated = false;
            observation_filter_enable = false;
            double graph_update_interval = ms_private_nh.param<double>("graph_update_interval", 3.0);
            double map_cloud_update_interval = ms_private_nh.param<double>("map_cloud_update_interval", 10.0);
            ms_optimization_timer = ms_mt_nh.createWallTimer(ros::WallDuration(graph_update_interval),
                                                             &hdl_graph_slam::MapServerNodelet::ms_optimization_timer_callback,
                                                             this);
            ms_map_publish_timer = ms_mt_nh.createWallTimer(ros::WallDuration(map_cloud_update_interval),
                                                            &hdl_graph_slam::MapServerNodelet::ms_map_points_publish_timer_callback,
                                                            this);

            for (int i = 0; i < 10; i++) {
                setInitialPosition[i] = true;
                initial_pose[i] = Eigen::Isometry3d::Identity();
            }
            ROS_INFO("Init done.");

        }

    private:

        /**
         * @brief received point clouds are pushed to #keyframe_queue
         * @param odom_msg
         * @param cloud_msg
         */
        void keyframe_msg_callback(const hdl_graph_slam::Keyframe_msg &keyframe_msg) {

            const ros::Time &stamp = keyframe_msg.header.stamp;
            newest_stamp = stamp;

            if (!new_run_published) {
                new_run_published = true;
                ROS_WARN("  ##########   NEW RUN STARTING   ##########  ");

                // Create the Header message
                std_msgs::Header header;
                header.seq = 0;
                header.stamp = stamp;
                header.frame_id = "new_run";  // Set the frame id

                // Publish the message
                new_run_pub.publish(header);
            }

            std::cout << "keyframe: " << keyframe_msg.id << " stamp: " << stamp << std::endl;

            if (std::find(registeredAgents.begin(), registeredAgents.end(), keyframe_msg.agent_no) ==
                registeredAgents.end()) {
                registeredAgents.push_back(keyframe_msg.agent_no);
            }

            pcl::PointCloud<PointT>::Ptr cloud_from_msg(new pcl::PointCloud<PointT>());

            pcl::MsgFieldMap field_map;
            pcl::createMapping<PointT>(keyframe_msg.cloud.fields, field_map);
            pcl::fromROSMsg(keyframe_msg.cloud, *cloud_from_msg);

            Eigen::Isometry3d kf_odom = Eigen::Translation3d(keyframe_msg.odom.position.x,
                                                             keyframe_msg.odom.position.y,
                                                             keyframe_msg.odom.position.z) *
                                        Eigen::Quaterniond(keyframe_msg.odom.orientation.w,
                                                           keyframe_msg.odom.orientation.x,
                                                           keyframe_msg.odom.orientation.y,
                                                           keyframe_msg.odom.orientation.z);

            if (isIdentity(kf_odom)) {
                ROS_WARN("  ##########   incoming odom is identity   ##########  ");
                std::cout << "Agent: " << keyframe_msg.agent_no << " id: " << keyframe_msg.id << std::endl;
                ROS_WARN("  ##########  ");
            }

            if (keyframes[keyframe_msg.agent_no].empty() && keyframe_queue[keyframe_msg.agent_no].empty()) {
                ROS_INFO("Setting initial pose");

                initial_pose[keyframe_msg.agent_no].translation().x() = keyframe_msg.initial.position.x;
                initial_pose[keyframe_msg.agent_no].translation().y() = keyframe_msg.initial.position.y;
                initial_pose[keyframe_msg.agent_no].translation().z() = keyframe_msg.initial.position.z;

                Eigen::Quaterniond q;
                q = Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitX())
                    * Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitY())
                    * Eigen::AngleAxisd(keyframe_msg.initial.orientation.z, Eigen::Vector3d::UnitZ());

                initial_orientation[keyframe_msg.agent_no] = q;
                initial_pose[keyframe_msg.agent_no].rotate(q.matrix());

            }


            KeyFrame::Ptr keyframe(
                    new KeyFrame(stamp, keyframe_msg.id, kf_odom,
                                 keyframe_msg.accum_distance, cloud_from_msg));
            keyframe_queue[keyframe_msg.agent_no].push_back(keyframe);

            ROS_INFO("New Keyframe received - Agent %i - ID: %i - Queue: %li", keyframe_msg.agent_no, keyframe_msg.id,
                     keyframe_queue[keyframe_msg.agent_no].size());

            // create an observation at the position of the keyframe
            int obs_id = keyframe_msg.id + 1000000;
            geometry_msgs::Pose empty_transform;

            DynObservation::Ptr obs(
                    new DynObservation(
                            obs_id,
                            stamp,
                            keyframe_msg.id,
                            keyframe_msg.agent_no,
                            10,
                            empty_transform
                    ));

            dynamic_observations.push_back(obs);
        }

        hdl_graph_slam::KeyFrame::Ptr findKeyframeById(int id) const {
            for (int a = 0; a < registeredAgents.size(); a++) {
                for (int i = 0; i < keyframes[registeredAgents[a]].size(); i++) {
                    if (keyframes[registeredAgents[a]][i]->keyframe_id == id) {
                        return keyframes[registeredAgents[a]][i];
                    }
                }
            }
            return nullptr;
        }

        bool isIdentity(const Eigen::Isometry3d &transform) {
            Eigen::Isometry3d identity = Eigen::Isometry3d::Identity();
            return transform.isApprox(identity);
        }

        std::tuple<int, int> findKeyframeIndexById(int id) const {
            std::cout << "findKeyframeIndexById: " << id << std::endl;
            for (int a = 0; a < registeredAgents.size(); a++) {
                for (int i = 0; i < keyframes[registeredAgents[a]].size(); i++) {
                    if (keyframes[registeredAgents[a]][i]->keyframe_id == id) {
                        return std::make_tuple(a, i);
                    }
                }
            }
            std::cout << "no keyframe index found for the ID: " << id << std::endl;
            return std::make_tuple(0, 999999);;
        }

        std::tuple<int, int> findKeyframeIndexByNodeId(int id) const {
            std::cout << "findKeyframeIndexByNodeId: " << id << std::endl;
            for (int a = 0; a < registeredAgents.size(); a++) {
                for (int i = 0; i < keyframes[registeredAgents[a]].size(); i++) {

                    if (keyframes[registeredAgents[a]][i]->node->id() == id) {
                        std::cout << "Found! Agent: " << a << " Index: " << i << std::endl;
                        return std::make_tuple(a, i);
                    }
                }
            }
            std::cout << "no keyframe index found for the node ID: " << id << std::endl;
            return std::make_tuple(0, 999999);;
        }

        int findNewKeyframeIndexById(int agent_no, int id) const {
            for (int i = 0; i < new_keyframes[agent_no].size(); i++) {
                if (new_keyframes[agent_no][i]->keyframe_id == id) {
                    return i;
                }
            }
            std::cout << "no new_keyframe index found for the ID: " << id << std::endl;
            return 0;;
        }

        /**
         * @brief received point clouds are pushed to #dynamic object queue
         * @param odom_msg
         * @param cloud_msg
         */
        void dynamic_object_msg_callback(const hdl_graph_slam::DynObservation_msg &observation_msg) {

            const ros::Time &stamp = observation_msg.obs_stamp;
            std::cout << "DYN: agent: " << observation_msg.agent_no << " object_id: " << observation_msg.id
                      << " stamp: " << stamp.toSec() << std::endl;

            newest_stamp = stamp;

            DynObservation::Ptr obs(
                    new DynObservation(
                            observation_msg.id,
                            stamp,
                            observation_msg.keyframe_id,
                            observation_msg.vehicle_id,
                            observation_msg.object_tag,
                            observation_msg.transform
                    ));

            int keyframe_id = observation_msg.keyframe_id;
            auto keyframe = findKeyframeById(keyframe_id);
            obs->keyframe = keyframe;

            obs->agent_no = observation_msg.agent_no;

            dynamic_observations.push_back(obs);

            for (const auto &obs_id: observation_ids) {
                // iterate through the already observed vehicle ids, if the current message-vehicle-id was
                // not observed yet, add it to the list
                if (obs_id == observation_msg.vehicle_id) {
                    return;
                }
            }

            observation_ids.push_back(observation_msg.vehicle_id);
        }

        /**
         * @brief received commands for the observation filter
         * @param observation_filter_msg
         */
        void observation_filter_callback(const hdl_graph_slam::ObservationFilter_msg &observation_filter_msg) {
            ROS_INFO("observation filter callback");
            observation_filter_enable = observation_filter_msg.enable;
            observation_filter_ids.clear();
            for (const auto &filter_id: observation_filter_msg.ids) {
                observation_filter_ids.push_back(filter_id);
                std::cout << observation_filter_ids.back() << ", " << std::endl;
            }

        }

        void ms_command_callback(const std_msgs::String::ConstPtr &msg) {

            std::string command = msg->data.c_str();
            ROS_INFO("Received Command: [%s]", msg->data.c_str());
            if (!command.compare("load_graph")) {
                ROS_INFO("Loading graph..");

                ROS_INFO("Successful");
                ms_optimization_timer_callback_func();
                ms_map_points_publish_timer_callback_func();
            } else if (!command.compare("pub_graph")) {
                ROS_INFO("Publishing graph");
                graph_slam->save("graph_agent");
            } else if (!command.compare("pub_graph")) {
                ROS_INFO("Publishing graph");
                graph_slam->save("graph_agent");
            }
        }

        /**
         * @brief this method adds all the keyframes in #keyframe_queue to the pose graph (odometry edges)
         * @return if true, at least one keyframe was added to the pose graph
         */
        bool flush_keyframe_queue() {
            ROS_INFO("flush keyframe");

            for (int a = 0; a < registeredAgents.size(); a++) {

                if (keyframe_queue[registeredAgents[a]].empty()) {
                    continue;
                }
                ROS_INFO("Agent %i: Keyframe queue size %li.", registeredAgents[a],
                         keyframe_queue[registeredAgents[a]].size());

                trans_odom2map_mutex.lock();
                Eigen::Isometry3d odom2map(initial_pose[registeredAgents[a]].matrix().cast<double>());
                trans_odom2map_mutex.unlock();

                int num_processed = 0;
                for (int i = 0; i < keyframe_queue[registeredAgents[a]].size(); i++) {
                    num_processed = i;

                    const auto &keyframe = keyframe_queue[registeredAgents[a]][i];
                    // new_keyframes will be tested later for loop closure
                    new_keyframes[registeredAgents[a]].push_back(keyframe);

                    // add pose node
                    Eigen::Isometry3d odom2map(initial_pose[registeredAgents[a]].matrix().cast<double>());
                    Eigen::Isometry3d odom = odom2map * keyframe->odom;

//                 if this is the first time visiting this place,
//                 set the first node to the initial pose and fix this one.
                    if (setInitialPosition[registeredAgents[a]]) {
                        keyframe->node = graph_slam->add_se3_node(initial_pose[registeredAgents[a]]);
                        keyframe->node->setFixed(true);
                        setInitialPosition[registeredAgents[a]] = false;
                    } else {
                        keyframe->node = graph_slam->add_se3_node(odom);
                        std::cout << "new keyframe node id: " << keyframe->node->id() << std::endl;
                    }
                    keyframe_hash[keyframe->stamp] = keyframe;

                    if (i == 0 && keyframes[registeredAgents[a]].empty()) {
                        continue;
                    }

                    // add edge between consecutive keyframes
                    const auto &prev_keyframe =
                            i == 0 ? keyframes[registeredAgents[a]].back() : keyframe_queue[registeredAgents[a]][i - 1];
                    Eigen::Isometry3d relative_pose = keyframe->odom.inverse() * prev_keyframe->odom;
                    Eigen::MatrixXd information = inf_calclator->calc_information_matrix(keyframe->cloud,
                                                                                         prev_keyframe->cloud,
                                                                                         relative_pose);
                    auto edge = graph_slam->add_se3_edge(keyframe->node, prev_keyframe->node, relative_pose,
                                                         information);
                }

                keyframe_queue[registeredAgents[a]].erase(keyframe_queue[registeredAgents[a]].begin(),
                                                          keyframe_queue[registeredAgents[a]].begin() + num_processed +
                                                          1);
            }
            return true;
        }

        void ms_map_points_publish_timer_callback(const ros::WallTimerEvent &event) {
            ms_map_points_publish_timer_callback_func();
        }

        /**
         * @brief generate map point cloud and publish it
         * @param event
         */
        void ms_map_points_publish_timer_callback_func() {
            ROS_INFO("MAP POINTS PUBLISH TIMER CALLBACK");
            if (!map_points_pub.getNumSubscribers()) {
                ROS_INFO("return: No subscribers");
                return;
            }

            if (!graph_updated) {
                ROS_INFO("return: Graph not updated");
                return;
            }

            std::vector <hdl_graph_slam::KeyFrameSnapshot::Ptr> snapshot;

            keyframes_snapshot_mutex.lock();
            snapshot = keyframes_snapshot;
            keyframes_snapshot_mutex.unlock();

            std::vector<int> reduce_classes = {7, 8, 9};

            auto cloud = map_cloud_generator->generate(snapshot, map_cloud_resolution_rough,
                                                       map_cloud_resolution_medium, map_cloud_resolution_fine,
                                                       reduce_classes);
            if (!cloud) {
                ROS_INFO("return: No cloud generated.");
                return;
            }

            ROS_WARN("  ##########   MAP GENERATION   ##########  ");

            double explored = 1.0 * cloud->size() / number_of_map_points_when_expored;

            std::cout << "Map points: " << cloud->size() << std::endl;
            std::cout << "Full map  : " << number_of_map_points_when_expored << std::endl;

            std::cout << "Map explored: " << explored * 100 << " % (" << explored << ")" << std::endl;

            // Create the Header message
            std_msgs::Header header_map;
            header_map.seq = explored * 10000;
            header_map.stamp = newest_stamp;
            header_map.frame_id = "map_explored";

            // Publish the message
            map_explored_pub.publish(header_map);

            if (exit_when_map_explored) {
                if (explored > 0.85) {

                    end_run_counter++;
                    std::cout << "Map explored! Counter = " << end_run_counter << std::endl;

                    if (end_run_counter > exit_iterations) {

                        new_run_published = true;
                        ROS_WARN("  ##########   MAP IS EXPLORED -> END RUN   ##########  ");

                        // Create the Header message
                        std_msgs::Header header;
                        header.seq = 0;
                        header.stamp = newest_stamp;
                        header.frame_id = "end_run";

                        // Publish the message
                        end_run_pub.publish(header);

                        exit(0);
                    }
                } else {
                    // if there is a keyframe, that is added to the map but not optized yet,
                    // it can be that the pcl is rotated, and so the number of points are rising
                    // very fast. in most cases, this keyframe is corrected within the next map.
                    // this counter is introduced, to only allow the run to end, if the number of points are
                    // consitently high enough to count the map as explored.

                    end_run_counter = 0;
                }
            }

            cloud->header.frame_id = map_frame_id;
            cloud->header.stamp = snapshot.back()->cloud->header.stamp;

            sensor_msgs::PointCloud2Ptr cloud_msg(new sensor_msgs::PointCloud2());
            pcl::toROSMsg(*cloud, *cloud_msg);

            map_points_pub.publish(cloud_msg);
            ROS_INFO("Map Points published.");
        }

        void ms_optimization_timer_callback(const ros::WallTimerEvent &event) {
            ms_optimization_timer_callback_func();
        }

        Eigen::Isometry3d getMergingEdgeRelativePose(int from_id, int to_id) {
            std::cout << "getMergingEdgeRelativePose from_id: " << from_id << ", to_id: " << to_id << std::endl;
            for (const auto &edge: merging_edges) {
                if (from_id == edge->from_node_id &&
                    to_id == edge->to_node_id) {
                    std::cout << "Found: " << *edge << std::endl;
                    return edge->relative_pose;
                } else if (from_id == edge->to_node_id &&
                           to_id == edge->from_node_id) {
                    std::cout << "Found: " << *edge << " (inverse)" << std::endl;
                    return edge->relative_pose.inverse();
                }
            }
            std::cout << "No edge found with from_id: " << from_id << " and to_id: " << to_id << std::endl;
            throw std::out_of_range("No Merging Edge with these ids found");
        }

        bool translationDifferenceExceedsOne(const Eigen::Isometry3d &iso1, const Eigen::Isometry3d &iso2) {
            // Subtract the translation vectors
            Eigen::Vector3d difference = iso1.translation() - iso2.translation();

            // Check if the absolute difference in x and y components is greater than 1
            return (std::abs(difference.x()) > 1.0) || (std::abs(difference.y()) > 1.0);
        }

        void node_merging() {

            std::cout << std::endl;
            ROS_INFO(" ----------------------");
            ROS_INFO(" ---- node merging ----");
            ROS_INFO(" ----------------------");
            std::cout << "nodes size: ";
            std::cout << graph_slam->graph->vertices().size();
            std::cout << " edges size:  ";
            std::cout << graph_slam->graph->edges().size() << std::endl;

            int loop_index = -1;
            for (const auto &loop: detected_loop_closures) {
                loop_index++;
                std::cout << "index: " << loop_index << std::endl;

                bool no_prev_keyframe = false;

                Eigen::Isometry3d lc_relative_pose(loop->relative_pose.cast<double>());

                /*
                check if the loop nodes can be merged
                the need to be neighter the first nor the last node of a trajectory

                loop->key1 is the newer keyframe that
                loop->key2 is the keyframe that existed earlier

                condition for the choosing of the keyframe that will be discarded:
                always discard the older keyframe.

                if discard_index = 0: this is a first point of an agent:
                  as long as the len(kf(of that agent)) > 1:
                      replace kf
                      you only have to change 1 edge
                  else:
                      skip
                if id = len(kf(of that agent): this is the latest point of the trajectory
                  skip
                else: this is a point within the trajectory
                  replace kf
                  change both edges

                edge replacement:
                  if an edge goes to a node, that is beside the new node:
                      skip the edge, just remove it
                */

                const int main_keyframe_id = loop->key1->keyframe_id;
                const int discard_keyframe_id = loop->key2->keyframe_id;

                int main_agent_index, main_keyframe_index;
                std::tie(main_agent_index, main_keyframe_index) = findKeyframeIndexById(main_keyframe_id);

                int discard_agent_index, discard_keyframe_index;
                std::tie(discard_agent_index, discard_keyframe_index) = findKeyframeIndexById(discard_keyframe_id);

                if ((discard_keyframe_index == 999999) || (main_keyframe_index == 999999)) {
                    detected_loop_closures.erase(detected_loop_closures.begin() + loop_index,
                                                 detected_loop_closures.begin() + loop_index + 1);
                    loop_index -= 1;
                    ROS_INFO("One of the keyframes not found in the correct queue. -> skip");
                    continue;
                }

                if (discard_keyframe_index + 1 == keyframes[discard_agent_index].size()) {
                    ROS_INFO("Discard_kf is the latest kf of this agent. -> skip");
                    continue;
                }

                if (keyframes[discard_agent_index].size() < 2) {
                    ROS_INFO("Agent has less than 2 kf. -> skip");
                    continue;
                }

                if (discard_keyframe_index == 0) {
                    ROS_INFO("This is the anchor node of a path. -> skip");
                    detected_loop_closures.erase(detected_loop_closures.begin() + loop_index,
                                                 detected_loop_closures.begin() + loop_index + 1);
                    continue;
                }

                std::cout << " key1: id: " << loop->key1->keyframe_id << "(node: " << loop->key1->node->id() << ")"
                          << std::endl
                          << " key2: id: " << loop->key2->keyframe_id << "(node: " << loop->key2->node->id() << ")"
                          << std::endl;

                auto main_keyframe = keyframes[main_agent_index][main_keyframe_index];

                std::cout << "main keyframe node id: " << main_keyframe->node->id() << std::endl;

                auto discard_keyframe = keyframes[discard_agent_index][discard_keyframe_index];

                std::cout << "discard keyframe node id: " << discard_keyframe->node->id() << std::endl;


                for (const auto &edge: graph_slam->graph->edges()) {
                    bool is_merging_edge = std::find_if(merging_edges.begin(), merging_edges.end(),
                                                        [&](const MergingEdge::Ptr obj) {
                                                            auto ret = (
                                                                    (obj->from_node_id == edge->vertices()[0]->id() &&
                                                                     obj->to_node_id == edge->vertices()[1]->id())
                                                                    ||
                                                                    (obj->from_node_id == edge->vertices()[1]->id() &&
                                                                     obj->to_node_id == edge->vertices()[0]->id())
                                                            );
                                                            return ret;
                                                        }) != merging_edges.end();

                    if (
                            (
                                    (edge->vertices()[0]->id() == discard_keyframe->node->id()) ||
                                    (edge->vertices()[1]->id() == discard_keyframe->node->id())
                            ) &&
                            (
                                    (edge->vertices()[0]->id() != main_keyframe->node->id()) &&
                                    (edge->vertices()[1]->id() != main_keyframe->node->id())
                            )
                            ) {

                        std::cout << std::endl;
                        std::cout << std::endl << "edge from: " << edge->vertices()[1]->id() << " -> "
                                  << edge->vertices()[0]->id()
                                  << " is a merging edge: ";
                        if (is_merging_edge) {
                            std::cout << "Yes" << std::endl;
                        } else {
                            std::cout << "No" << std::endl;
                        }

                        // add edge between consecutive keyframes

                        int prev_discard_agent_index, prev_discard_keyframe_index;
                        bool discard_kf_is_first_in_edge;

                        if (edge->vertices()[0]->id() == discard_keyframe->node->id()) {
                            std::tie(prev_discard_agent_index, prev_discard_keyframe_index) = findKeyframeIndexByNodeId(
                                    edge->vertices()[1]->id());
                            discard_kf_is_first_in_edge = true;
                        } else {
                            std::tie(prev_discard_agent_index, prev_discard_keyframe_index) = findKeyframeIndexByNodeId(
                                    edge->vertices()[0]->id());
                            discard_kf_is_first_in_edge = false;
                        }
                        std::cout << "--> discard_prev: " << prev_discard_agent_index << ", "
                                  << prev_discard_keyframe_index << std::endl;

                        if (prev_discard_keyframe_index == 999999) {
                            no_prev_keyframe = true;
                            std::cout << "No prev discard keyframe index found. Node id: " << edge->vertices()[0]->id()
                                      << std::endl;
                            continue;
                        }

                        auto discard_prev_keyframe = keyframes[prev_discard_agent_index][prev_discard_keyframe_index];

                        if (is_merging_edge) {
                            // if the edge would result in a egde on the own nearby trajectory, dont add it

                            // check if the kf is part of the same agent
                            if ((main_keyframe->keyframe_id / 1000000) ==
                                (discard_prev_keyframe->keyframe_id / 1000000)) {

                                // check if the keyframe is near to the main keyframe
                                if (discard_prev_keyframe->keyframe_id > (main_keyframe->keyframe_id - 10)) {
                                    ROS_INFO("one merging edge would have been to the same trajectory");
                                    continue;
                                }
                            }
                        }

                        if (isIdentity(discard_keyframe->odom)) {
                            ROS_WARN("  ##########   discard_keyframe odom is identity   ##########  ");
                        }

                        if (isIdentity(discard_prev_keyframe->odom)) {
                            ROS_WARN("  ##########   discard_prev_keyframe keyframe odom is identity   ##########  ");
                        }

                        Eigen::Isometry3d kf_relative_pose = Eigen::Isometry3d::Identity();

                        g2o::VertexSE3 *main_node = main_keyframe->node;
                        g2o::VertexSE3 *disc_node = discard_keyframe->node;
                        g2o::VertexSE3 *prev_disc_node = discard_prev_keyframe->node;

                        Eigen::Isometry3d kf_relative_pose_estimate = Eigen::Isometry3d::Identity();

                        if (!discard_kf_is_first_in_edge) {
                            kf_relative_pose_estimate = prev_disc_node->estimate().inverse() * disc_node->estimate();
                        } else {
                            kf_relative_pose_estimate = disc_node->estimate().inverse() * prev_disc_node->estimate();
                        }


                        if (is_merging_edge) {
                            Eigen::Isometry3d merging_edge_transform = Eigen::Isometry3d::Identity();
                            try {
                                merging_edge_transform = getMergingEdgeRelativePose(edge->vertices()[0]->id(),
                                                                                    edge->vertices()[1]->id());
                            }
                            catch (const std::exception &e) {
                                ROS_WARN("exception case");
                            }

                            ROS_WARN("GOT RELATIVE POSE FROM MERGING EDGE OBJECT");
                            ROS_WARN("GOT RELATIVE POSE FROM MERGING EDGE OBJECT");
                            ROS_WARN("GOT RELATIVE POSE FROM MERGING EDGE OBJECT");


                            std::cout << "kf rel pose from merging edge: " << std::endl
                                      << merging_edge_transform.matrix() << std::endl;

                            std::cout << "kf rel pose from estimates: " << std::endl
                                      << kf_relative_pose_estimate.matrix() << std::endl;


                            kf_relative_pose.linear() = merging_edge_transform.linear();
                            kf_relative_pose.translation() = merging_edge_transform.translation();


                        } else {
                            std::cout << "Normal edge: " << std::endl;
                            kf_relative_pose = discard_keyframe->odom.inverse() * discard_prev_keyframe->odom;
                            std::cout << "discard kf odom: " << std::endl << discard_keyframe->odom.matrix()
                                      << std::endl;
                            std::cout << "prev discard kf odom: " << std::endl << discard_prev_keyframe->odom.matrix()
                                      << std::endl;
                        }

                        std::cout << "kf rel pose: " << std::endl << kf_relative_pose.matrix() << std::endl;

                        if (translationDifferenceExceedsOne(kf_relative_pose, kf_relative_pose_estimate) &&
                            is_merging_edge) {
                            std::cout << "differnce of one element (x or y) was exceeding 1 -> taking the estimate"
                                      << std::endl;
                            kf_relative_pose = kf_relative_pose_estimate;
                            std::cout << "after exchanging kf rel pose: " << std::endl << kf_relative_pose.matrix()
                                      << std::endl;
                        }

                        double kf_distance_poses = kf_relative_pose.translation().norm();
                        std::cout << "kf rel pose length (rel poses): " << kf_distance_poses << std::endl;

                        if (kf_distance_poses > 20.0) {
                            kf_relative_pose.translation() = kf_relative_pose_estimate.translation();

                            kf_distance_poses = kf_relative_pose.translation().norm();
                            std::cout << "new kf rel pose length (rel poses): " << kf_distance_poses << std::endl;
                            std::cout << "kf estimates matrix: " << std::endl << kf_relative_pose_estimate.matrix()
                                      << std::endl;
                            std::cout << "kf resulting matrix: " << std::endl << kf_relative_pose.matrix() << std::endl;
                        }

                        double lc_distance_poses = lc_relative_pose.translation().norm();
                        std::cout << "lc rel pose length (rel poses): " << lc_distance_poses << std::endl;
                        std::cout << "lc rel pose: " << std::endl << lc_relative_pose.matrix() << std::endl;

                        Eigen::Isometry3d relative_pose = lc_relative_pose * kf_relative_pose;

                        Eigen::Quaterniond rel_pos_q(relative_pose.rotation());
                        Eigen::Quaterniond est_pos_q(
                                (main_node->estimate().inverse() * prev_disc_node->estimate()).rotation());
                        Eigen::Quaterniond qRelativeRotation = rel_pos_q.inverse() * est_pos_q;

                        // Compute the angle difference (in radians)
                        double angleDifference = 2 * std::acos(std::min(std::max(qRelativeRotation.w(), -1.0), 1.0));

                        std::cout << "Angle difference in poses: " << angleDifference << std::endl;

                        if (std::abs(angleDifference) > 0.5) {
                            std::cout << "Angle difference too high, taking estimation" << std::endl;
                            // this is a major source of errors.
                            // i investigated a lot into this error, but i cannot solve it at the moment
                            // the workaround is to skip the generation of this additional merging edge
                            // for stability this case must be solved in the future.
                            continue;

                            kf_relative_pose = kf_relative_pose_estimate;

                            relative_pose = lc_relative_pose * kf_relative_pose;
                            Eigen::Quaterniond new_rel_pos_q(relative_pose.rotation());
                            Eigen::Quaterniond new_est_pos_q(
                                    (main_node->estimate().inverse() * prev_disc_node->estimate()).rotation());
                            qRelativeRotation = new_rel_pos_q.inverse() * new_est_pos_q;

                            // Compute the angle difference (in radians)
                            angleDifference = 2 * std::acos(std::min(std::max(qRelativeRotation.w(), -1.0), 1.0));

                            std::cout << "Angle difference in poses: " << angleDifference << std::endl;
                        }

                        // check the length of this edge, to avoid bad edges
                        double distance_merge_edge = (prev_disc_node->estimate().translation() -
                                                      main_node->estimate().translation()).norm();
                        double distance_rel_poses = relative_pose.translation().norm();


                        std::cout << "Merging edge length (estimates): " << distance_merge_edge << std::endl;
                        std::cout << "Merging edge length (rel poses): " << distance_rel_poses << std::endl;
                        std::cout << "Length difference in poses: "
                                  << std::abs(distance_rel_poses - distance_merge_edge) << std::endl;


                        if (std::abs(distance_merge_edge - distance_rel_poses) > 2.0) {
                            ROS_WARN(" DISTANCE DIFFERENCE ");
                            continue;
                            if (!is_merging_edge) {

                                relative_pose = lc_relative_pose * kf_relative_pose;
                                distance_rel_poses = relative_pose.translation().norm();
                                std::cout << "New rel pose: " << std::endl << relative_pose.matrix() << std::endl;
                                std::cout << "New Merging length (rel poses) : " << distance_rel_poses << std::endl;
                            } else {
                                std::cout << "No change because it was a merging edge." << std::endl;
                            }
                        }

                        std::cout << std::endl;

                        Eigen::MatrixXd information = inf_calclator->calc_information_matrix(main_keyframe->cloud,
                                                                                             discard_prev_keyframe->cloud,
                                                                                             relative_pose);
                        if (is_merging_edge) {
                            information.topLeftCorner(3, 3).array() /= 100;
                        }

                        auto graph_edge = graph_slam->add_se3_edge(main_keyframe->node, discard_prev_keyframe->node,
                                                                   relative_pose,
                                                                   information);

                        MergingEdge::Ptr m_edge(
                                new MergingEdge(main_keyframe->stamp, main_keyframe_id, main_keyframe->node->id(),
                                                discard_prev_keyframe->keyframe_id, discard_prev_keyframe->node->id(),
                                                relative_pose));
                        std::cout << *m_edge << std::endl;

                        merging_edges.push_back(m_edge);
                    }
                }

                if (no_prev_keyframe) {
                    continue;
                }

                std::cout << std::endl;

                std::cout << "############ remove vertex with id: " << discard_keyframe->node->id() << " ############"
                          << std::endl;
                graph_slam->graph->removeVertex(discard_keyframe->node, false);
                std::cout << "node removal done. " << std::endl << std::endl;

                // node merging
                pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud_1(new pcl::PointCloud <pcl::PointXYZINormal>);
                pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud_in(new pcl::PointCloud <pcl::PointXYZINormal>);
                pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud_out(new pcl::PointCloud<pcl::PointXYZINormal>());

                // get cloud from the main_kf
                *cloud_1 = *(loop->key1->cloud);
                // get cloud from discard_kf
                *cloud_in = *(loop->key2->cloud);

                // transform discard_kf cloud to the main_kf cloud and add it
                geometry_msgs::Transform transform;
                tf::transformEigenToMsg(lc_relative_pose, transform);
                tf::Transform tf_old_kf_to_new_kf;
                tf_old_kf_to_new_kf.setOrigin(
                        tf::Vector3(transform.translation.x, transform.translation.y, transform.translation.z));
                tf_old_kf_to_new_kf.setRotation(
                        tf::Quaternion(transform.rotation.x, transform.rotation.y, transform.rotation.z,
                                       transform.rotation.w));
                pcl_ros::transformPointCloud(*cloud_in, *cloud_out, tf_old_kf_to_new_kf);

                float max_percent = 0.4;
                int max_time_sec = 10 * 60;
                int min_time_sec = 0.5 * 60;

                double diff_sec = (loop->key1->stamp - loop->key2->stamp).toSec();

                float transfer_percent = 0.0;
                if (diff_sec < min_time_sec) {
                    transfer_percent = max_percent;
                } else if (diff_sec >= max_time_sec) {
                    transfer_percent = 0.0;
                } else {
                    double diff_scaled = (diff_sec - 30) / 570;
                    transfer_percent = diff_scaled * max_percent;
                }

                std::cout << transfer_percent << std::endl;

                pcl::PointCloud<pcl::PointXYZINormal>::Ptr merged_cloud(new pcl::PointCloud <pcl::PointXYZINormal>);
                if (transfer_percent < 0.01) {
                    *merged_cloud = *cloud_1;
                    std::cout << "Only taking the new cloud. Discard the old cloud." << std::endl;
                } else {
                    // Randomly sample a percentage of points from cloud_out
                    pcl::RandomSample <pcl::PointXYZINormal> sample_old;
                    sample_old.setInputCloud(cloud_out);
                    sample_old.setSample(transfer_percent * cloud_out->size());
                    pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud_out_reduced(
                            new pcl::PointCloud <pcl::PointXYZINormal>);
                    sample_old.filter(*cloud_out_reduced);


                    pcl::RandomSample <pcl::PointXYZINormal> sample_new;
                    sample_new.setInputCloud(cloud_1);
                    sample_new.setSample((1.0 - transfer_percent) * cloud_out->size());
                    pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud_1_reduced(
                            new pcl::PointCloud <pcl::PointXYZINormal>);
                    sample_new.filter(*cloud_1_reduced);

                    std::cout << "Taking " << (1.0 - transfer_percent) * 100 << "% of new pointcloud:"
                              << cloud_1->size()
                              << " -> "
                              << cloud_1_reduced->size() << std::endl;
                    std::cout << "Taking " << transfer_percent * 100 << "% of old pointcloud:" << cloud_out->size()
                              << " -> "
                              << cloud_out_reduced->size() << std::endl;

                    pcl::concatenate(*cloud_1_reduced, *cloud_out_reduced, *merged_cloud);
                }

                // create new kf object with the new merged cloud
                KeyFrame::Ptr new_keyframe(
                        new KeyFrame(main_keyframe->stamp, main_keyframe->keyframe_id, main_keyframe->odom,
                                     main_keyframe->accum_distance,
                                     merged_cloud));
                new_keyframe->node = main_keyframe->node;
                keyframes[main_agent_index][main_keyframe_index] = new_keyframe;

                // erase the discard_kf from the vector
                keyframes[discard_agent_index].erase(
                        keyframes[discard_agent_index].begin() + discard_keyframe_index,
                        keyframes[discard_agent_index].begin() + discard_keyframe_index + 1);

                // remap the dynamic observations from the discard_kf to the main_kf
                for (const auto &dyn_obs: dynamic_observations) {

                    if (dyn_obs->keyframe_id == discard_keyframe_id) {
                        // the dyn_obs belongs to the discard_kf

                        // reset the id to the new kf
                        dyn_obs->keyframe_id = main_keyframe_id;
                        dyn_obs->keyframe = main_keyframe;

                        Eigen::Isometry3d observation_transform;
                        tf::poseMsgToEigen(dyn_obs->transformFromKeyframe, observation_transform);
                        Eigen::Isometry3d transformed_pose = lc_relative_pose * observation_transform;

                        geometry_msgs::Pose new_kf_to_obs_pose_msg;

                        tf::poseEigenToMsg(transformed_pose, new_kf_to_obs_pose_msg);

                        dyn_obs->transformFromKeyframe = new_kf_to_obs_pose_msg;
                    }
                }

                // remove loop from the list as the nodes have been merged
                detected_loop_closures.erase(detected_loop_closures.begin() + loop_index,
                                             detected_loop_closures.begin() + loop_index + 1);
                loop_index -= 1;
            }

            std::cout << "after merging " << std::endl;
            std::cout << "nodes size: ";
            std::cout << graph_slam->graph->vertices().size();
            std::cout << " edges size:  ";
            std::cout << graph_slam->graph->edges().size() << std::endl;

            ROS_INFO("node merging done.");

            graph_slam->initialze_graph();
        }

        /**
         * @brief this methods adds all the data in the queues to the pose graph, and then optimizes the pose graph
         * @param event
         */
        void ms_optimization_timer_callback_func() {
            ROS_INFO("OPTIMIZATION TIMER CALLBACK");

            std::lock_guard <std::mutex> lock(main_thread_mutex);

            // add keyframes and floor coeffs in the queues to the pose graph
            bool keyframe_updated = flush_keyframe_queue();

            if (!keyframe_updated) {
                ROS_INFO("Optimization Callback: Return: Keyframes not updated.");
                return;
            }


            // loop detection
            std::vector <KeyFrame::Ptr> all_keyframes;

            for (int a = 0; a < registeredAgents.size(); a++) {
                for (int i = 0; i < keyframes[registeredAgents[a]].size(); i++) {
                    all_keyframes.push_back(keyframes[registeredAgents[a]][i]);
                }
            }

            int num_iterations1 = ms_private_nh.param<int>("g2o_solver_num_iterations", 1024);
            graph_slam->optimize(num_iterations1);

            for (int a = 0; a < registeredAgents.size(); a++) {
                ROS_INFO("Loop Detection Agent %i",
                         registeredAgents[a]);
                std::vector <hdl_graph_slam::Loop::Ptr> loops = loop_detector->detect(all_keyframes,
                                                                                      new_keyframes[registeredAgents[a]],
                                                                                      *graph_slam,
                                                                                      registeredAgents[a]);
                ROS_INFO("Loop Detection Agent %i: %li loops found - all keyframes: %li - new: %li",
                         registeredAgents[a], loops.size(), all_keyframes.size(),
                         new_keyframes[registeredAgents[a]].size());

                for (const auto &loop: loops) {
                    ROS_INFO("\n ----- LOOP FOUND ----- ");
                    Eigen::Isometry3d relpose(loop->relative_pose.cast<double>());
                    std::cout << "Keyframe " << loop->key1->keyframe_id << " -> " << loop->key2->keyframe_id
                              << std::endl;
                    Eigen::MatrixXd information_matrix = inf_calclator->calc_information_matrix(loop->key1->cloud,
                                                                                                loop->key2->cloud,
                                                                                                relpose);
                    std::cout << "info: " << std::endl << information_matrix << std::endl;
                    information_matrix.topLeftCorner(3, 3).array() *= 100;

                    std::cout << "relpos graph: " << std::endl << relpose.matrix() << std::endl;

                    auto edge = graph_slam->add_se3_edge(loop->key1->node, loop->key2->node, relpose,
                                                         information_matrix);

                    // check if there is already a loop towards this target keyframe
                    bool kf_is_already_detected =
                            std::find_if(detected_loop_closures.begin(), detected_loop_closures.end(),
                                         [&](const hdl_graph_slam::Loop::Ptr detected_loop) {
                                             std::cout << "loop target: " << loop->key2->keyframe_id
                                                       << " detect_target: " << detected_loop->key2->keyframe_id
                                                       << std::endl;
                                             auto ret = (loop->key2->keyframe_id == detected_loop->key2->keyframe_id);
                                             return ret;
                                         }) != detected_loop_closures.end();

                    if (!kf_is_already_detected) {
                        detected_loop_closures.push_back(loop);
                    }
                }
                ROS_INFO("Optimization done.");


            }

            for (int a = 0; a < registeredAgents.size(); a++) {
                if (new_keyframes[a].empty()) {
                    continue;
                }

                std::copy(new_keyframes[registeredAgents[a]].begin(), new_keyframes[registeredAgents[a]].end(),
                          std::back_inserter(keyframes[registeredAgents[a]]));
                new_keyframes[registeredAgents[a]].clear();
            }

            node_merging();

            std::vector <KeyFrameSnapshot::Ptr> snapshot(all_keyframes.size());
            std::transform(all_keyframes.begin(), all_keyframes.end(), snapshot.begin(),
                           [=](const KeyFrame::Ptr &k) {
                               auto s = std::make_shared<KeyFrameSnapshot>(k);
                               return s;
                           });

            keyframes_snapshot_mutex.lock();
            keyframes_snapshot.swap(snapshot);
            keyframes_snapshot_mutex.unlock();
            graph_updated = true;

            groupObservations();

            auto markers = create_marker_array(newest_stamp);
            ROS_INFO("viz back.");
            try {
                markers_pub.publish(markers);
            }
            catch (...) {
                std::cout << "publishing the markers failed" << std::endl;
            }

            ROS_INFO("pub done.");
            ROS_INFO("Optimization and viz done.");
        }

        float value_boundary(float value) const {
            if ((value < -1000) || (value > 1000)) {
                return 0.0;
            }
            return value;
        }

        bool match_id(DynObservation::Ptr obs, int search_id) const {
            return obs->id == search_id;
        }

        hdl_graph_slam::DynObservation::Ptr findObservationById(int search_id) const {

            // Find the first DynObservation::Ptr in dynamic_observations with the given id.
            auto it = std::find_if(dynamic_observations.begin(), dynamic_observations.end(),
                                   [&](DynObservation::Ptr obs) { return match_id(obs, search_id); });


            if (it != dynamic_observations.end()) {
                // Object with id x was found, and 'it' points to its position in the vector
                // You can access the object using '*it', or get its index with 'std::distance(dynamic_observations.begin(), it)'
                return *it;
            } else {
                // Object with id x was not found in the vector
                return nullptr;
            }
        }

        void removeEdge(g2o::SparseOptimizer &optimizer, int vertexId1, int vertexId2) const {
            std::cout << "1" << std::endl;
            // Retrieve the vertices using their IDs
            g2o::OptimizableGraph::Vertex *v1 = optimizer.vertex(vertexId1);
            std::cout << "2" << std::endl;
            g2o::OptimizableGraph::Vertex *v2 = optimizer.vertex(vertexId2);
            std::cout << "3" << std::endl;

            if (!v1 || !v2) {
                std::cerr << "One of the vertices not found in the graph." << std::endl;
                return;
            }
            std::cout << "4" << std::endl;

            g2o::EdgeSE3 *edgeToRemove = nullptr;
            std::cout << "5" << std::endl;

            // Iterate through the edges connected to the first vertex
            for (const auto &edge: v1->edges()) {
                g2o::EdgeSE3 *e = dynamic_cast<g2o::EdgeSE3 *>(edge);
                std::cout << "6" << std::endl;

                if (e) {
                    std::cout << "7" << std::endl;
                    // Check if the other vertex connected by the edge is v2
                    if (e->vertices()[0] == v1 && e->vertices()[1] == v2) {
                        std::cout << "8" << std::endl;
                        edgeToRemove = e;
                        std::cout << "9" << std::endl;
                        break;
                    }
                }
            }
            std::cout << "10" << std::endl;

            if (edgeToRemove) {
                std::cout << "11" << std::endl;
                // Remove the edge from the graph
                optimizer.removeEdge(edgeToRemove);
                std::cout << "12" << std::endl;

                // Optionally, update the vertices' connectivity
                v1->edges().erase(edgeToRemove);
                std::cout << "13" << std::endl;
                v2->edges().erase(edgeToRemove);
                std::cout << "14" << std::endl;

                // Delete the edge to free memory
                std::cout << "15" << std::endl;

                // Optionally, reinitialize the optimization
                optimizer.initializeOptimization();
                std::cout << "16" << std::endl;
            }
        }

        /**
         * @brief create visualization marker
         * @param stamp
         * @return
         */
        visualization_msgs::MarkerArray create_marker_array(const ros::Time &stamp) const {

            ROS_INFO("Viz start. --------------------------------------");
            visualization_msgs::MarkerArray markers;
            markers.markers.resize(3);


            // node markers
            visualization_msgs::Marker &traj_marker = markers.markers[0];
            traj_marker.header.frame_id = map_frame_id;
            traj_marker.header.stamp = stamp;
            traj_marker.ns = "nodes";
            traj_marker.id = 0;
            traj_marker.type = visualization_msgs::Marker::SPHERE_LIST;

            traj_marker.pose.orientation.w = 1.0;
            traj_marker.scale.x = traj_marker.scale.y = traj_marker.scale.z = 2.0;

            std::vector <KeyFrame::Ptr> all_keyframes;
            for (int a = 0; a < registeredAgents.size(); a++) {
                for (int i = 0; i < keyframes[registeredAgents[a]].size(); i++) {
                    all_keyframes.push_back(keyframes[registeredAgents[a]][i]);
                }
            }

            traj_marker.points.resize(all_keyframes.size());
            traj_marker.colors.resize(all_keyframes.size());

            ROS_INFO("marker");
            for (int i = 0; i < all_keyframes.size(); i++) {
                Eigen::Vector3d pos = all_keyframes[i]->node->estimate().translation();

                traj_marker.points[i].x = pos.x();
                traj_marker.points[i].y = pos.y();
                traj_marker.points[i].z = pos.z() + height_offset_keyframes;

                double p = static_cast<double>(i) / all_keyframes.size();
                traj_marker.colors[i].r = 0.5;
                traj_marker.colors[i].g = 1.0;
                traj_marker.colors[i].b = 0.0;
                traj_marker.colors[i].a = 1.0;
            }

//             edge markers
            visualization_msgs::Marker &edge_marker = markers.markers[1];
            edge_marker.header.frame_id = map_frame_id;
            edge_marker.header.stamp = stamp;
            edge_marker.ns = "edges";
            edge_marker.id = 2;
            edge_marker.type = visualization_msgs::Marker::LINE_LIST;

            edge_marker.pose.orientation.w = 1.0;
            edge_marker.scale.x = 0.3;

            std::cout << "edges size:  ";
            std::cout << graph_slam->graph->edges().size();
            std::cout << " nodes size: ";
            std::cout << graph_slam->graph->vertices().size();
            std::cout << std::endl;


            edge_marker.points.resize(graph_slam->graph->edges().size() * 2);
            edge_marker.colors.resize(graph_slam->graph->edges().size() * 2);

            auto edge_itr = graph_slam->graph->edges().begin();
            for (int i = 0; edge_itr != graph_slam->graph->edges().end(); edge_itr++, i++) {
                g2o::HyperGraph::Edge *edge = *edge_itr;
                g2o::EdgeSE3 *edge_se3 = dynamic_cast<g2o::EdgeSE3 *>(edge);

                g2o::VertexSE3 *v1 = dynamic_cast<g2o::VertexSE3 *>(edge_se3->vertices()[0]);
                g2o::VertexSE3 *v2 = dynamic_cast<g2o::VertexSE3 *>(edge_se3->vertices()[1]);
                Eigen::Vector3d pt1 = v1->estimate().translation();
                Eigen::Vector3d pt2 = v2->estimate().translation();

                // check the length of this edge, to avoid bad edges
                double distance = (pt2 - pt1).norm();
                if (distance > 40.0) {

                    ROS_WARN("  ##########   BAD EDGE FOUND   ##########  ");

                    std::cout << "Distance: " << distance << std::endl;
                    std::cout << std::endl;

                    auto kf1_ids = findKeyframeIndexByNodeId(v1->id());
                    int agent_no1 = get<0>(kf1_ids);
                    int kf_index1 = get<1>(kf1_ids);
                    std::cout << "node1: " << v1->id() << std::endl;
                    std::cout << "agent: " << registeredAgents[agent_no1] << std::endl;
                    std::cout << "kf_index: " << kf_index1 << std::endl;
                    std::cout << "keyframe_id: " << keyframes[registeredAgents[agent_no1]][kf_index1]->keyframe_id
                              << std::endl;

                    std::cout << std::endl;

                    auto kf2_ids = findKeyframeIndexByNodeId(v2->id());
                    int agent_no2 = get<0>(kf2_ids);
                    int kf_index2 = get<1>(kf2_ids);
                    std::cout << "node2: " << v2->id() << std::endl;
                    std::cout << "agent: " << registeredAgents[agent_no2] << std::endl;
                    std::cout << "kf_index: " << kf_index2 << std::endl;
                    std::cout << "keyframe_id: " << keyframes[registeredAgents[agent_no2]][kf_index2]->keyframe_id
                              << std::endl;

                    ROS_WARN("  ##########   BAD EDGE FOUND   ##########  ");

                    std::cout << "remove func" << std::endl;

                    removeEdge(static_cast<g2o::SparseOptimizer &>(*graph_slam->graph), v1->id(), v2->id());

                    std::cout << "remove func end" << std::endl;

                    visualization_msgs::MarkerArray empty_markers;
                    return empty_markers;
                }

                edge_marker.points[i * 2].x = pt1.x();
                edge_marker.points[i * 2].y = pt1.y();
                edge_marker.points[i * 2].z = pt1.z();
                edge_marker.points[i * 2 + 1].x = pt2.x();
                edge_marker.points[i * 2 + 1].y = pt2.y();
                edge_marker.points[i * 2 + 1].z = pt2.z();

                double p1 = static_cast<double>(v1->id()) / graph_slam->graph->vertices().size();
                double p2 = static_cast<double>(v2->id()) / graph_slam->graph->vertices().size();

                edge_marker.colors[i * 2].r = 0.5;
                edge_marker.colors[i * 2].g = 0.9;
                edge_marker.colors[i * 2].a = 1.0;
                edge_marker.colors[i * 2 + 1].r = 0.5;
                edge_marker.colors[i * 2 + 1].g = 0.9;
                edge_marker.colors[i * 2 + 1].a = 1.0;

                edge_marker.points[i * 2].z += height_offset_keyframes;
                edge_marker.points[i * 2 + 1].z += height_offset_keyframes;
            }

            ROS_INFO("observations");

            visualization_msgs::Marker &dyn_observation_marker = markers.markers[2];
            dyn_observation_marker.header = traj_marker.header;
            dyn_observation_marker.ns = "dynamic_observations";
            dyn_observation_marker.id = 1;
            dyn_observation_marker.type = visualization_msgs::Marker::CUBE_LIST;

            dyn_observation_marker.pose.orientation.w = 1.0;
            dyn_observation_marker.scale.x = dyn_observation_marker.scale.y = dyn_observation_marker.scale.z = 0.75;

            std::vector <DynObservation::Ptr> dynamic_observations_snapshot(dynamic_observations);

            std::cout << "size original: " << dynamic_observations.size() << std::endl;
            std::cout << "size copy: " << dynamic_observations_snapshot.size() << std::endl;


            dyn_observation_marker.points.resize(dynamic_observations_snapshot.size());
            dyn_observation_marker.colors.resize(dynamic_observations_snapshot.size());

            int min_id = 10000;
            int max_id = 0;

            for (const auto &obs: dynamic_observations_snapshot) {
                if (obs->vehicle_id > 10) {
                    min_id = std::min(min_id, obs->vehicle_id);
                }
                max_id = std::max(max_id, obs->vehicle_id);
            }

            hdl_graph_slam::DynObservationArray_msg observation_array_msg;
            observation_array_msg.header.stamp = newest_stamp;
            observation_array_msg.header.frame_id = map_frame_id;
            observation_array_msg.observations.resize(dynamic_observations_snapshot.size());

            hdl_graph_slam::DynObservationArray_msg observation_array_msg_for_metrics;
            observation_array_msg_for_metrics.header.stamp = newest_stamp;
            observation_array_msg_for_metrics.header.frame_id = map_frame_id;
            std::deque <geometry_msgs::PoseStamped> metric_poses;


            if (dynamic_observations_snapshot.size() > dyn_observation_marker.points.size()) {
                ROS_WARN("Not enough space in marker for all dynamic observations");
                return markers;
            }

            for (int i = 0; i < dynamic_observations_snapshot.size(); i++) {
                Eigen::Isometry3d observation_transform;
                tf::poseMsgToEigen(dynamic_observations_snapshot[i]->transformFromKeyframe, observation_transform);

                Eigen::Isometry3d keyframe_pose;
                if (!dynamic_observations_snapshot[i]->keyframe) {
                    DynObservation::Ptr obs = dynamic_observations_snapshot[i];
                    const int keyframe_id = obs->keyframe_id;
                    auto keyframe = findKeyframeById(keyframe_id);

                    if (keyframe) {
                        dynamic_observations_snapshot[i]->keyframe = keyframe;
                    } else {
                        std::cout << "NOT FOUND keyframe (nullptr again)" << std::endl;
                        continue;
                    }
                    std::cout << "keyframe " << keyframe_id << " = "
                              << dynamic_observations_snapshot[i]->keyframe->keyframe_id << " for observation "
                              << obs->id << " found!" << std::endl;
                }

                keyframe_pose = dynamic_observations_snapshot[i]->keyframe->node->estimate();

                Eigen::Isometry3d obs_pose = keyframe_pose * observation_transform;
                Eigen::Vector3d pos = obs_pose.translation();

                dyn_observation_marker.points[i].x = value_boundary(pos.x());
                dyn_observation_marker.points[i].y = value_boundary(pos.y());
                dyn_observation_marker.points[i].z = value_boundary(pos.z());

                geometry_msgs::PoseStamped single_observation_for_array;
                single_observation_for_array.pose.position.x = value_boundary(pos.x());
                single_observation_for_array.pose.position.y = value_boundary(pos.y());
                single_observation_for_array.pose.position.z = value_boundary(pos.z());
                single_observation_for_array.header.stamp = dynamic_observations_snapshot[i]->stamp;

                double p = 1.0 * (dynamic_observations_snapshot[i]->vehicle_id - min_id) / (max_id - min_id);

                dyn_observation_marker.colors[i].a = 1.0;
                dyn_observation_marker.colors[i].g = p;
                dyn_observation_marker.colors[i].b = 1.0 - p;
                dyn_observation_marker.colors[i].r = 1.0 - p;

                if (dynamic_observations[i]->object_tag == 4) {
                    // show pedestrians as almost black
                    dyn_observation_marker.colors[i].r = 0.1;
                    dyn_observation_marker.colors[i].g = 0.1;
                    dyn_observation_marker.colors[i].b = 0.1;

                    single_observation_for_array.header.seq = 999999;

                } else if (dynamic_observations[i]->object_tag == 10) {
                    // vehicles
                    single_observation_for_array.header.seq = dynamic_observations[i]->vehicle_id;

                    if (dynamic_observations[i]->vehicle_id < 10) {
                        // this is the observation created by a keyframe of an agent
                        // publish this extra on a topic to create slam metrics as these observations
                        // represent the estimated poses of the agents

                        metric_poses.push_back(single_observation_for_array);
                    }
                } else {
                    // error case = white sign
                    dyn_observation_marker.colors[i].r = 1.0;
                    dyn_observation_marker.colors[i].g = 1.0;
                    dyn_observation_marker.colors[i].b = 1.0;
                }

                observation_array_msg.observations[i] = single_observation_for_array;
            }
            ROS_INFO("obs array done");

            observation_array_pub.publish(observation_array_msg);
            ROS_INFO("obs array published");


            observation_array_msg_for_metrics.observations.resize(metric_poses.size());
            std::cout << "metric poses size: " << metric_poses.size() << std::endl;

            for (int i = 0; i < metric_poses.size(); i++) {
                observation_array_msg_for_metrics.observations[i] = metric_poses.at(i);;
            }

            keyframe_metric_pub.publish(observation_array_msg_for_metrics);
            ROS_INFO("keyframe metric msg published");


            geometry_msgs::PointStamped merge_metrics_msg;
            merge_metrics_msg.point.x = metric_poses.size();
            merge_metrics_msg.point.y = graph_slam->graph->vertices().size();
            merge_metrics_msg.header.stamp = newest_stamp;

            merge_metric_pub.publish(merge_metrics_msg);
            ROS_INFO("merge metric msg published");


            ROS_INFO("Viz done. --------------------------------------");
            return markers;
        }

        double angleBetweenVectors(const Eigen::Vector3d &ab, const Eigen::Vector3d &ac) {
            double cosTheta = ab.dot(ac) / (ab.norm() * ac.norm());
            double theta = std::acos(cosTheta);
            double degrees = theta * 180 / M_PI;

            if (std::abs(degrees) > std::abs(degrees - 180)) {
                degrees = abs(degrees - 180);
            }
            return degrees;
        }

        struct VehicleObservations {
            int vehicle_id;
            std::vector <DynObservation::Ptr> observations;
        };

        std::vector <VehicleObservations> groupObservationsByVehicleId() {
            std::unordered_map<int, VehicleObservations> vehicleObservationsMap;

            for (auto it = dynamic_observations.begin(); it != dynamic_observations.end();) {
                const auto &observation = *it;

                if (vehicleObservationsMap.find(observation->vehicle_id) == vehicleObservationsMap.end()) {
                    vehicleObservationsMap[observation->vehicle_id] = {observation->vehicle_id, {}};
                }

                VehicleObservations &currentVehicleObservations = vehicleObservationsMap[observation->vehicle_id];
                bool shouldAdd = true;

                if (!currentVehicleObservations.observations.empty()) {
                    const auto &prevObservation = currentVehicleObservations.observations.back();

                    double timeDifference = (observation->stamp - prevObservation->stamp).toSec();

                    if (!observation->keyframe) {
                        std::cout << "observation keyframe is empty. ID: " << observation->id << " KF: "
                                  << observation->keyframe_id << std::endl;
                        ++it;
                        continue;
                    }

                    if (!prevObservation->keyframe) {
                        std::cout << "prevObservaion keyframe is empty. ID: " << prevObservation->id << " KF: "
                                  << prevObservation->keyframe_id << std::endl;
                        ++it;
                        continue;
                    }

                    Eigen::Isometry3d observation_transform;
                    tf::poseMsgToEigen(observation->transformFromKeyframe, observation_transform);
                    Eigen::Isometry3d obs_pose = observation->keyframe->node->estimate() * observation_transform;

                    Eigen::Isometry3d prev_observation_transform;
                    tf::poseMsgToEigen(prevObservation->transformFromKeyframe, prev_observation_transform);
                    Eigen::Isometry3d prev_obs_pose =
                            prevObservation->keyframe->node->estimate() * prev_observation_transform;
                    double distance = (obs_pose.translation() - prev_obs_pose.translation()).norm();

                    bool angle_criterium = true;

                    if (currentVehicleObservations.observations.size() > 2) {
                        auto iter = currentVehicleObservations.observations.end();
                        --iter;
                        --iter;
                        auto secondLast_obs = *iter;

                        if (!secondLast_obs->keyframe) {
                            std::cout << "secondlast keyframe is empty." << std::endl;
                            ++it;
                            continue;
                        }

                        Eigen::Isometry3d secondlast_observation_transform;
                        tf::poseMsgToEigen(secondLast_obs->transformFromKeyframe, secondlast_observation_transform);
                        Eigen::Isometry3d secondlast_obs_pose =
                                secondLast_obs->keyframe->node->estimate() * secondlast_observation_transform;

                        Eigen::Vector3d prev_vector = prev_obs_pose.translation() - secondlast_obs_pose.translation();
                        Eigen::Vector3d current_vector = obs_pose.translation() - prev_obs_pose.translation();

                        double angle = angleBetweenVectors(prev_vector, current_vector);

                        if (angle > 25.0) {
                            angle_criterium = false;
                        }
                    }

                    if ((timeDifference * distance < 0.1) || (distance < 0.5) && angle_criterium) {
                        shouldAdd = false;
                        it = dynamic_observations.erase(it);
                        std::cout << "Observation removed: id: " << observation->id << " kf-id:"
                                  << observation->keyframe_id << " distance: " << distance << " timeDiff: "
                                  << timeDifference << std::endl;
                        continue;
                    }

                }

                if (shouldAdd) {
                    currentVehicleObservations.observations.push_back(observation);
                }
                ++it;
            }

            std::cout << "loop out" << std::endl;

            std::vector <VehicleObservations> groupedObservations;
            groupedObservations.reserve(vehicleObservationsMap.size());
            for (const auto &entry: vehicleObservationsMap) {
                groupedObservations.push_back(entry.second);
            }

            std::cout << "done" << std::endl;

            return groupedObservations;
        }

        int groupObservations() {
            int old_size = dynamic_observations.size();
            groupObservationsByVehicleId();

            std::cout << "DYN OBS: removed " << old_size - dynamic_observations.size() << " observations. "
                      << std::endl;


            return 0;
        }

    private:
        // ROS
        ros::NodeHandle ms_nh;
        ros::NodeHandle ms_mt_nh;
        ros::NodeHandle ms_private_nh;
        ros::WallTimer ms_optimization_timer;
        ros::WallTimer ms_map_publish_timer;

        std::unique_ptr <message_filters::Subscriber<nav_msgs::Odometry>> odom_sub;
        std::unique_ptr <message_filters::Subscriber<sensor_msgs::PointCloud2>> cloud_sub;
        std::unique_ptr <message_filters::Synchronizer<ApproxSyncPolicy>> sync;
        ros::Subscriber ms_command_sub;
        ros::Subscriber keyframe_msg_sub;
        ros::Subscriber dynamic_observation_msg_sub;
        ros::Subscriber observation_filter_sub;

        double height_offset_keyframes;

        bool observation_filter_enable;
        bool new_run_published;
        bool setInitialPosition[10];
        bool initial_guess[10];
        Eigen::Isometry3d initial_pose[10];
        Eigen::Quaterniond initial_orientation[10];
        std::deque<int> registeredAgents;

        ros::Publisher markers_pub;
        ros::Publisher observation_array_pub;
        ros::Publisher keyframe_metric_pub;
        ros::Publisher merge_metric_pub;

        std::mutex trans_odom2map_mutex;
        std::vector <Eigen::Matrix4f> trans_odom2map;
        ros::Publisher odom2map_pub;

        std::string map_server_topic;
        std::string map_frame_id;
        std::string agent_no;

        std::string agent_keyframe_topic;
        std::string observation_filter_topic;
        std::string dynamic_observation_topic;

        ros::Publisher read_until_pub;
        ros::Publisher map_points_pub;
        ros::Publisher dynamic_object_box_pub;

        ros::Publisher merge_test_c1;
        ros::Publisher merge_test_c2;
        ros::Publisher merge_test_cT;
        ros::Publisher merge_test_final;
        ros::Publisher merge_test_kf_old;
        ros::Publisher merge_test_kf_new;
        ros::Publisher new_run_pub;
        ros::Publisher end_run_pub;
        ros::Publisher map_explored_pub;

        tf::TransformListener tf_listener;

        ros::ServiceServer dump_service_server;
        ros::ServiceServer save_map_service_server;


        // for map cloud generation
        std::atomic_bool graph_updated;
        double map_cloud_resolution_rough;
        double map_cloud_resolution_medium;
        double map_cloud_resolution_fine;
        std::mutex keyframes_snapshot_mutex;
        std::vector <hdl_graph_slam::KeyFrameSnapshot::Ptr> keyframes_snapshot;
        std::unique_ptr <hdl_graph_slam::MapCloudGenerator> map_cloud_generator;

        // graph slam
        // all the below members must be accessed after locking main_thread_mutex
        std::mutex main_thread_mutex;

        int max_keyframes_per_update;
        std::deque <hdl_graph_slam::KeyFrame::Ptr> new_keyframes[10];
        std::deque <hdl_graph_slam::Loop::Ptr> detected_loop_closures;
        std::deque <hdl_graph_slam::KeyFrame::Ptr> keyframe_queue[10];

        ros::Time newest_stamp;
        bool first_agent;

        bool exit_when_map_explored;
        double number_of_map_points_when_expored;
        int exit_iterations;

        int end_run_counter;

        g2o::VertexSE3 *anchor_node;
        g2o::EdgeSE3 *anchor_edge;
        g2o::VertexPlane *floor_plane_node;
        std::vector <hdl_graph_slam::KeyFrame::Ptr> keyframes[10];
        std::vector <DynObservation::Ptr> dynamic_observations;
        std::vector<int> observation_ids;
        std::vector<int> observation_filter_ids;
        std::vector <MergingEdge::Ptr> merging_edges;
        std::unordered_map <ros::Time, hdl_graph_slam::KeyFrame::Ptr, RosTimeHash> keyframe_hash;

        std::unique_ptr <hdl_graph_slam::GraphSLAM> graph_slam;
        std::unique_ptr <hdl_graph_slam::LoopDetector> loop_detector;
        std::unique_ptr <hdl_graph_slam::KeyframeUpdater> keyframe_updater;

        std::unique_ptr <hdl_graph_slam::InformationMatrixCalculator> inf_calclator;
    };

}  // namespace hdl_graph_slam

PLUGINLIB_EXPORT_CLASS(hdl_graph_slam::MapServerNodelet, nodelet::Nodelet
)
