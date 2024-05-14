// SPDX-License-Identifier: BSD-2-Clause

#ifndef LOOP_DETECTOR_HPP
#define LOOP_DETECTOR_HPP

#include <boost/format.hpp>
#include <hdl_graph_slam/keyframe.hpp>
#include <hdl_graph_slam/registrations.hpp>
#include <hdl_graph_slam/graph_slam.hpp>

#include <g2o/types/slam3d/vertex_se3.h>

#include <hdl_graph_slam/custom_point_types.hpp>

#include <pcl/filters/conditional_removal.h>
#include <pcl/filters/passthrough.h>

namespace hdl_graph_slam {

    struct Loop {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        using Ptr = std::shared_ptr<Loop>;

        Loop(const KeyFrame::Ptr &key1, const KeyFrame::Ptr &key2, const Eigen::Matrix4f &relpose) : key1(key1),
                                                                                                     key2(key2),
                                                                                                     relative_pose(
                                                                                                             relpose) {}

    public:
        KeyFrame::Ptr key1;
        KeyFrame::Ptr key2;
        Eigen::Matrix4f relative_pose;
    };

/**
 * @brief this class finds loops by scam matching and adds them to the pose graph
 */
    class LoopDetector {
    public:
        typedef pcl::PointXYZINormal PointT;

        /**
         * @brief constructor
         * @param pnh
         */
        LoopDetector(ros::NodeHandle &pnh) {
            distance_thresh = pnh.param<double>("distance_thresh", 5.0);
            accum_distance_thresh = pnh.param<double>("accum_distance_thresh", 8.0);
            distance_from_last_edge_thresh = pnh.param<double>("min_edge_interval", 5.0);

            fitness_score_max_range = pnh.param<double>("fitness_score_max_range", std::numeric_limits<double>::max());
            fitness_score_thresh = pnh.param<double>("fitness_score_thresh", 0.5);

            registration = select_registration_method(pnh);

            for (int i = 0; i < 10; i++) {
                last_edge_accum_distance[i] = 0.0;
            }
        }

        /**
         * @brief detect loops and add them to the pose graph
         * @param keyframes       keyframes
         * @param new_keyframes   newly registered keyframes
         * @param graph_slam      pose graph
         */
        std::vector <Loop::Ptr>
        detect(const std::vector <KeyFrame::Ptr> &keyframes, const std::deque <KeyFrame::Ptr> &new_keyframes,
               hdl_graph_slam::GraphSLAM &graph_slam, int agent_no) {
            std::vector <Loop::Ptr> detected_loops;
            for (const auto &new_keyframe: new_keyframes) {
                auto candidates = find_candidates(keyframes, new_keyframe, agent_no);
//                ROS_INFO("Loop Candidates: %li", candidates.size());
                auto loop = matching(candidates, new_keyframe, graph_slam, agent_no);
                if (loop) {
                    detected_loops.push_back(loop);
                }
            }

            return detected_loops;
        }

        double get_distance_thresh() const {
            return distance_thresh;
        }

    private:
        /**
         * @brief find loop candidates. A detected loop begins at one of #keyframes and ends at #new_keyframe
         * @param keyframes      candidate keyframes of loop start
         * @param new_keyframe   loop end keyframe
         * @return loop candidates
         */
        std::vector <KeyFrame::Ptr>
        find_candidates(const std::vector <KeyFrame::Ptr> &keyframes, const KeyFrame::Ptr &new_keyframe,
                        int agent_no) const {
            // too close to the last registered loop edge
            if (new_keyframe->accum_distance - last_edge_accum_distance[agent_no] <
                distance_from_last_edge_thresh) {
                ROS_INFO("find candidates: return: distance from last edge thresh");
                std::cout << new_keyframe->accum_distance << " / " << last_edge_accum_distance[agent_no] << std::endl;
                return std::vector<KeyFrame::Ptr>();
            }

            std::vector <KeyFrame::Ptr> candidates;
            candidates.reserve(32);

            for (const auto &k: keyframes) {
////       traveled distance between keyframes is too small
//      this is not needed on the server node, as the accum distance of agents can be differnt
//      if(new_keyframe->accum_distance - k->accum_distance < accum_distance_thresh) {
//          ROS_INFO("find candidates: return: accum distance");
//        continue;
//      }
//                std::cout << "agent no: " << agent_no << std::endl;
//                std::cout << "kf id: " << k->keyframe_id << std::endl;
//                std::cout << "modu: " << k->keyframe_id/1000 << std::endl;
//                std::cout << agent_no << " : " << k->keyframe_id << " : " << k->keyframe_id / 1000000 << std::endl;
                if (k->keyframe_id / 1000000 == agent_no) {
//                    std::cout << "candidate: same agent: acum base: " << new_keyframe->accum_distance << " cand: "
//                              << k->accum_distance << std::endl;
                    if (new_keyframe->accum_distance - k->accum_distance < 25) {
//                        std::cout << "candidate: same agent. too near" << std::endl;
                        continue;
                    }
                }

                const auto &pos1 = k->node->estimate().translation();
                const auto &pos2 = new_keyframe->node->estimate().translation();


//       estimated distance between keyframes is too small
                double dist = (pos1.head<2>() - pos2.head<2>()).norm();
//       std::cout << "distance btween frames: "<< boost::format("%.3f") % dist << std::endl;
                if (dist > distance_thresh) {
//                    ROS_INFO("find candidates: return: distance thresh");
                    continue;
                }

                candidates.push_back(k);
            }

            return candidates;
        }

        pcl::PointCloud<PointT>::Ptr createMutableCloudCopy(const pcl::PointCloud<PointT>::ConstPtr& const_cloud) {
            pcl::PointCloud<PointT>::Ptr mutable_cloud(new pcl::PointCloud<PointT>);
            *mutable_cloud = *const_cloud;
            return mutable_cloud;
        }

        pcl::PointCloud<PointT>::Ptr filterPointCloud(const pcl::PointCloud<PointT>::Ptr& src_cloud) {

            const std::vector<int> classes_for_high_resolution = {1, 2, 3, 5, 6, 7, 11, 12, 13, 14, 17, 18, 19, 20};

            // Create a new point cloud to hold the filtered results
            pcl::PointCloud<PointT>::Ptr cloud_of_small_objects(new pcl::PointCloud <PointT>);

            // Create a new condition
            pcl::ConditionOr<PointT>::Ptr range_cond(new pcl::ConditionOr<PointT>());

            // Add a comparison to the condition for each class
            for (const auto &class_id: classes_for_high_resolution) {
                range_cond->addComparison(pcl::FieldComparison<PointT>::ConstPtr(
                        new pcl::FieldComparison<PointT>("intensity", pcl::ComparisonOps::EQ, class_id)));
            }

            // Create the ConditionalRemoval filter and set it up
            pcl::ConditionalRemoval <PointT> condrem;
            condrem.setCondition(range_cond);
            condrem.setInputCloud(src_cloud);

            // Apply the filter
            condrem.filter(*cloud_of_small_objects);

            return cloud_of_small_objects;
        }

        /**
         * @brief To validate a loop candidate this function applies a scan matching between keyframes consisting the loop. If they are matched well, the loop is added to the pose graph
         * @param candidate_keyframes  candidate keyframes of loop start
         * @param new_keyframe         loop end keyframe
         * @param graph_slam           graph slam
         */
        Loop::Ptr
        matching(const std::vector <KeyFrame::Ptr> &candidate_keyframes, const KeyFrame::Ptr &new_keyframe,
                 hdl_graph_slam::GraphSLAM &graph_slam, int agent_no) {
            if (candidate_keyframes.empty()) {
                return nullptr;
            }



//            registration->setInputSource(new_keyframe->cloud);
//            auto mutable_cloud = createMutableCloudCopy(new_keyframe->cloud);
            registration->setInputTarget(new_keyframe->cloud);

//            registration->setInputTarget(filterPointCloud(new_keyframe->cloud));

            double best_score = 1000.0;
//            double best_score = std::numeric_limits<double>::max();
            KeyFrame::Ptr best_matched;
            Eigen::Matrix4f relative_pose;

            std::cout << std::endl;
            std::cout << "--- loop detection ---" << std::endl;
            std::cout << "agent: " << agent_no << std::endl;
//            if (initial_guess) {
//                std::cout << "mode: initial" << std::endl;
//            } else {
//                std::cout << "mode: first loop already found" << std::endl;
//            }
            std::cout << "num_candidates: " << candidate_keyframes.size() << std::endl;
            std::cout << "matching" << std::flush;
            auto t1 = ros::Time::now();
            Eigen::Isometry3d new_keyframe_estimate = new_keyframe->node->estimate();
            new_keyframe_estimate.linear() = Eigen::Quaterniond(
                    new_keyframe_estimate.linear()).normalized().toRotationMatrix();


            pcl::PointCloud<PointT>::Ptr aligned(new pcl::PointCloud<PointT>());
            for (const auto &candidate: candidate_keyframes) {

//                auto mutable_candidate_cloud = createMutableCloudCopy(candidate->cloud);

                registration->setInputSource(candidate->cloud);

                std::cout << std::endl << "inpcut cloud: " << candidate->cloud->width << std::endl;

//                if (candidate->cloud->width < 5000) {
//                    std::cout << "skipped small cloud." << std::endl;
//                    continue;
//                }

                Eigen::Isometry3d candidate_estimate = candidate->node->estimate();
                candidate_estimate.linear() = Eigen::Quaterniond(
                        candidate_estimate.linear()).normalized().toRotationMatrix();
                Eigen::Matrix4f guess = (new_keyframe_estimate.inverse() *
                                         candidate_estimate).matrix().cast<float>();
                guess(2, 3) = 0.0;

//                if (initial_guess) {
//                    guess = Eigen::Matrix4f::Zero();
//                    guess.setIdentity();
//                }

                registration->align(*aligned, guess);

//                std::cout << "." << std::flush;

//                if (initial_guess || 1) {
//                    if (!registration->hasConverged()) {
//                        std::cout << "-";
//                        continue;
//                    }

//                    double score1 = registration->getFitnessScore(fitness_score_max_range);
//
//                    std::cout << "naiv score: " << score1 << std::endl;
//
//                    Eigen::Matrix4f relative_guess = registration->getFinalTransformation();
//                    std::cout << relative_guess << std::endl;
//                    double dx = 0.0;
//                    dx = relative_guess.block<3, 1>(0, 3).norm();
//
//                    std::cout << "distance accepted." << std::endl;
//                    registration->align(*aligned, relative_guess);
//                }
                std::cout << new_keyframe->keyframe_id << " -> "
                          << candidate->keyframe_id;

                double score = registration->getFitnessScore(fitness_score_max_range);

                std::cout << " score: " << score << std::endl;

//                double x = relative_pose.block<3, 1>(0, 3).norm();
//                std::cout << "dx: " << x << std::endl;
//                if (x < 0.1 || x > 5.0) {
//                    continue;
//                }

                if (!registration->hasConverged() || score > best_score) {
                    std::cout << "not converged." << std::endl;
                    continue;
                }
//                if (x == 0.0) {
//                    std::cout << "dx = 0" << std::endl;
//                    continue;
//                }
                best_score = score;
                best_matched = candidate;
                relative_pose = registration->getFinalTransformation();

//                    if (best_score < fitness_score_thresh) {
//                        std::cout << "loop found..." << std::endl;
//                        detected_loops.push_back(std::make_shared<Loop>(new_keyframe, best_matched, relative_pose));
//                    }
            }

            auto t2 = ros::Time::now();
//            std::cout << " done" << std::endl;
            std::cout << "best_score: " << boost::format("%.3f") % best_score << "    time: "
                      << boost::format("%.3f") % (t2 - t1).toSec() << "[sec]" << std::endl;

            if (best_score > fitness_score_thresh) {
                std::cout << "loop not found..." << std::endl;
                return nullptr;
            }

            std::cout << std::endl << " --------------------------- " << std::endl << " # # # LOOP FOUND # # # "
                      << std::endl << " --------------------------- " << std::endl;
            std::cout << "relpose: " << std::endl << relative_pose.block<3, 1>(0, 3) << " - "
                      << Eigen::Quaternionf(relative_pose.block<3, 3>(0, 0)).coeffs() << std::endl;

            double dx = relative_pose.block<3, 1>(0, 3).norm();
            std::cout << "dx: " << dx << " ";

            last_edge_accum_distance[agent_no] = new_keyframe->accum_distance;

            return std::make_shared<Loop>(new_keyframe, best_matched, relative_pose);
        }

    private:
        double distance_thresh;                 // estimated distance between keyframes consisting a loop must be less than this distance
        double accum_distance_thresh;           // traveled distance between ...
        double distance_from_last_edge_thresh;  // a new loop edge must far from the last one at least this distance

        double fitness_score_max_range;  // maximum allowable distance between corresponding points
        double fitness_score_thresh;     // threshold for scan matching

        double last_edge_accum_distance[10];

        pcl::Registration<PointT, PointT>::Ptr registration;
    };

}  // namespace hdl_graph_slam

#endif  // LOOP_DETECTOR_HPP
