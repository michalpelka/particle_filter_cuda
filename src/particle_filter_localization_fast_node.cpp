#include <GL/freeglut.h>

//PCL
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
//#include <pcl/point_representation.h>
#include <pcl/io/pcd_io.h>
//#include <pcl/filters/voxel_grid.h>
//#include <pcl/filters/filter.h>
//#include <pcl/features/normal_3d.h>
//#include <pcl/registration/transforms.h>
//#include <pcl/registration/ndt.h>
#include <pcl/console/parse.h>
//#include <pcl/registration/icp.h>
//#include <pcl/common/time.h>
//#include <pcl/filters/voxel_grid.h>
//#include <pcl/filters/statistical_outlier_removal.h>
//#include <pcl/PCLPointCloud2.h>

#include "basicFunctions.h"
#include "particle_filter_fast.h"

//common
#include "data_model.hpp"
#include "point_types.h"
#include "cudaWrapper.h"

// ros

#include <ros/ros.h>
#include <pcl_ros/point_cloud.h>
#include <tf/transform_listener.h>
#include <tf_conversions/tf_eigen.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <nav_msgs/Odometry.h>


typedef struct scan_with_odo
{
	pcl::PointCloud<pcl::PointXYZ>  scan;

	Eigen::Affine3f odo;
}scan_with_odo_t;



class nodePF
{
public:
	nodePF():
	n("~")
	{
		lastOdometry = Eigen::Affine3f::Identity();

		std::string topic_laser3D;
		n.param<std::string>("topic_laser3D", topic_laser3D, "/velodyne_points");
		ROS_INFO("param topic_laser3D: '%s'", topic_laser3D.c_str());

		//subInput   = n.subscribe(topic_laser3D,1, &nodeSlam::scanCallback, this);
		subInput   			= n.subscribe(topic_laser3D,1, &nodePF::scanCallback, this);
		subPose   			= n.subscribe("/initialpose",1, &nodePF::initalPoseCallback, this);

		bestPointcloudPub 	= n.advertise<pcl::PointCloud<Semantic::PointXYZL> >("bestPointCloud",1);
		mapPointcloudPub	= n.advertise<pcl::PointCloud<Semantic::PointXYZL> >("mapPointCloud",1);;
		particlePub			= n.advertise<geometry_msgs::PoseArray>("particles",1);
		odometryPub			= n.advertise<nav_msgs::Odometry>("odometry",1);


		//motion_model_max_angle = 10.0f;
		n.param<float>("motion_model_max_angle", motion_model_max_angle, 5.0f);

		//motion_model_max_translation = 0.5f;
		n.param<float>("motion_model_max_translation", motion_model_max_translation, 0.3f);

		//max_particle_size = 1000;
		n.param<int>("max_particle_size", max_particle_size, 200);


		//max_particle_size_kidnapped_robot = 10000;
		n.param<int>("max_particle_size_kidnapped_robot", max_particle_size_kidnapped_robot, 200);

		//distance_above_Z = 1.0f;
		n.param<float>("distance_above_Z", distance_above_Z, 1);


		//rgd_resolution = 1.0f;
		n.param<float>("rgd_resolution", rgd_resolution, 1);

		//cuda_device = 0;
		n.param<int>("cuda_device", cuda_device, 0);

		//max_number_considered_in_INNER_bucket = 5;
		n.param<int>("max_number_considered_in_INNER_bucket", max_number_considered_in_INNER_bucket, 5);

		//max_number_considered_in_OUTER_bucket = 5;
		n.param<int>("max_number_considered_in_OUTER_bucket", max_number_considered_in_OUTER_bucket, 5);

		//overlap_threshold = 0.01f;
		n.param<float>("overlap_threshold", overlap_threshold, 0.01f);

		//propability_threshold = 0.1;
		n.param<float>("propability_threshold", propability_threshold, 0.1);

		//rgd_2D_res = 5.0f;
		n.param<float>("rgd_2D_res", rgd_2D_res, 5.0f);

		std::string fnMapClassified;
		std::string fnMap;
		n.param<std::string>("mapClassified", fnMapClassified,"");
		n.param<std::string>("map", fnMap, "");

		if (fnMapClassified.size() != 0 && fnMap.size() !=0)
		{
			ROS_FATAL("only one of parameter mapClassified and map can be set.");
			exit(-1);
		}
		if (fnMapClassified.size() == 0 && fnMap.size() ==0)
		{
			ROS_FATAL("no map given. closing");
			exit(-1);
		}

		// load PCD and classify
		if (fnMap.size() != 0)
		{
			pcl::PointCloud<pcl::PointXYZ> pointcloud_notlabeled;
			ROS_INFO("Loading pcd file : %s", fnMap.c_str());
			if(pcl::io::loadPCDFile(fnMap,pointcloud_notlabeled) == -1)
			{
				ROS_FATAL("cannot load PCD!");
				exit(-1);
			}
			CPointcloudClassifier classifier(0);
			classifier.classify(pointcloud_notlabeled, point_cloud_semantic_map);
		}
		// load PCD only
		if (fnMapClassified.size() != 0)
		{
			ROS_INFO("Loading pcd file : %s", fnMapClassified.c_str());
			if(pcl::io::loadPCDFile(fnMapClassified,point_cloud_semantic_map) == -1)
			{
				ROS_FATAL("cannot load PCD!");
				exit(-1);
			}
		}


		for(size_t i = 0 ; i < point_cloud_semantic_map.size(); i++)
		{
			if(point_cloud_semantic_map[i].label == FLOOR_GROUND)
			{
				pcl::PointXYZ p;
				p.x = point_cloud_semantic_map[i].x;
				p.y = point_cloud_semantic_map[i].y;
				p.z = point_cloud_semantic_map[i].z;
				point_cloud_semantic_map_label_floor_ground.push_back(p);
			}
		}

		ROS_INFO("point_cloud_semantic_map_label_floor_ground size : %d", point_cloud_semantic_map_label_floor_ground.size());


		if(!particle_filter.init(cuda_device,
				motion_model_max_angle,
				motion_model_max_translation,
				max_particle_size,
				max_particle_size_kidnapped_robot,
				distance_above_Z,
				rgd_resolution,
				max_number_considered_in_INNER_bucket,
				max_number_considered_in_OUTER_bucket,
				overlap_threshold,
				propability_threshold,
				rgd_2D_res))
		{
			ROS_FATAL ("problem with particle_filter.init() exit(-1)");
			exit(-1);
		}
		if(!particle_filter.setGroundPointsFromMap(point_cloud_semantic_map_label_floor_ground))
		{
			ROS_FATAL("problem with particle_filter.setGroundPointsFromMap() exit(-1)");
			exit(-1);
		}

		particle_filter.genParticlesKidnappedRobot();

		if(!particle_filter.copyReferenceModelToGPU(point_cloud_semantic_map))
		{
			ROS_FATAL("problem with particle_filter.copyReferenceModelToGPU(point_cloud_semantic_map)  exit(-1)");
			exit(-1);
		}
		if(!particle_filter.computeRGD())
		{
			ROS_FATAL("problem with particle_filter.computeRGD() exit(-1)");
			exit(-1);
		}

		ros::spin();

	}
private:
	void scanCallback(pcl::PointCloud<pcl::PointXYZ> scan);
	void initalPoseCallback(geometry_msgs::PoseWithCovarianceStamped pose);
	ros::NodeHandle n;


	ros::Subscriber subInput;
	ros::Subscriber subPose;

	ros::Publisher bestPointcloudPub;
	ros::Publisher mapPointcloudPub;
	ros::Publisher particlePub;
	ros::Publisher odometryPub;


	pcl::PointCloud<Semantic::PointXYZL> point_cloud_semantic_map;
	pcl::PointCloud<pcl::PointXYZ> point_cloud_semantic_map_label_floor_ground;

	pcl::PointCloud<Semantic::PointXYZL> winning_point_cloud;

	tf::TransformListener tf_listener;
	Eigen::Affine3f lastOdometry;

	float motion_model_max_angle;
	float motion_model_max_translation;
	int max_particle_size;
	int max_particle_size_kidnapped_robot;
	float distance_above_Z;
	float rgd_resolution;
	int cuda_device;
	int max_number_considered_in_INNER_bucket;
	int max_number_considered_in_OUTER_bucket;
	float overlap_threshold;
	float propability_threshold;
	float rgd_2D_res;
	CParticleFilterFast particle_filter;

};

void nodePF::initalPoseCallback(geometry_msgs::PoseWithCovarianceStamped pose)
{
	ROS_INFO("get initial pose");
	particle_filter.setPose(pose);
}
void nodePF::scanCallback(pcl::PointCloud<pcl::PointXYZ> scan)
{
	try
		{

		//tf_listener->lookupTransform(frame_global, frame_robot, msg->header.stamp,
		//		position_current);
		tf::StampedTransform position_current;
		tf_listener.lookupTransform("odom", /*frame_robot*/ scan.header.frame_id,ros::Time(0), position_current);

		Eigen::Affine3d dm = Eigen::Affine3d::Identity();
		tf::transformTFToEigen (position_current, dm);

		Eigen::Affine3f currentOdometry = dm.cast<float>();


		clock_t begin_time;
		double computation_time;
		begin_time = clock();

		static int counter = 1;

		Eigen::Affine3f odometryIncrement =lastOdometry.inverse() * currentOdometry;//= vscan_with_odo[counter-1].odo.inverse() * vscan_with_odo[counter].odo;

		lastOdometry = currentOdometry;
		pcl::PointCloud<Semantic::PointXYZL> labeled;

		CPointcloudClassifier classifier (0);
		classifier.classify(scan,labeled);

		if(!particle_filter.copyCurrentScanToGPU(labeled))
		{
			std::cout << "problem with cuda_nn.copyCurrentScanToGPU(current_scan) return" << std::endl;
			return;
		}

		if(!particle_filter.update())return;

		Eigen::Affine3f winM = particle_filter.getWinningParticle();

		//if(show_winning_point_cloud)
		{
			winning_point_cloud = labeled;
			transformPointCloud(winning_point_cloud, winning_point_cloud, winM);
		}
		particle_filter.prediction(odometryIncrement);


		counter++;


		computation_time=(double)( clock () - begin_time ) /  CLOCKS_PER_SEC;
		std::cout << "particle filter singleIteration computation_time: " << computation_time << " counter: "<< counter <"\n";

		point_cloud_semantic_map.header.frame_id = "map";
		mapPointcloudPub.publish(point_cloud_semantic_map);

		winning_point_cloud.header.frame_id = "map";
		bestPointcloudPub.publish(winning_point_cloud);

		geometry_msgs::PoseArray particles = particle_filter.getPoseArray();
		particlePub.publish(particles);



		nav_msgs::Odometry odoMsg = particle_filter.getOdom();
		odoMsg.child_frame_id=scan.header.frame_id;
		odometryPub.publish(odoMsg);



	}
	catch (tf::TransformException &ex)
	{
		ROS_ERROR_THROTTLE(100,"%s", ex.what());
	}
}


int main(int argc, char** argv)
{
	ros::init(argc, argv, "mini_slam");
	nodePF();
}




/// runs clasifier on given pcl::PointXYZ pointcloud, returns Semantic::PointXYZL

