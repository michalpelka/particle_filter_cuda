#ifndef __PARTICLE_FILTER_FAST__
#define __PARTICLE_FILTER_FAST__

//#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
//#include <pcl_ros/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>

//common
#include "point_types.h"
#include "cudaFunctions.h"

#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <nav_msgs/Odometry.h>

#include <cudaWrapper.h>
// helper fuunction to classify pointcloud
class CPointcloudClassifier
{
public:
	CPointcloudClassifier(int cudaDevice = 0,
			float normal_vectors_search_radius = 1.0f,
			float curvature_threshold = 0.0f,
			float ground_Z_coordinate_threshold = 1,
			int number_of_points_needed_for_plane_threshold =1,
			int max_number_considered_in_INNER_bucket = 100,
			int max_number_considered_in_OUTER_bucket = 100);

	void classify (pcl::PointCloud<pcl::PointXYZ>in, pcl::PointCloud<Semantic::PointXYZL>&out);
private:
	int cudaDevice;
	float normal_vectors_search_radius;
	float curvature_threshold;
	float ground_Z_coordinate_threshold;
	int number_of_points_needed_for_plane_threshold;
	int max_number_considered_in_INNER_bucket;
	int max_number_considered_in_OUTER_bucket;
};

class CParticleFilterFast
{
public:
	typedef struct particle_state
	{
		Eigen::Affine3f matrix;
		float overlap;
	}particle_state_t;

	typedef struct particle
	{
		bool isOverlapOK;
		float W;
		float nW;
		std::vector<particle_state_t> v_particle_states;
	}particle_t;

	CParticleFilterFast();
	~CParticleFilterFast();

	bool init(int cuda_device,
			float _motion_model_max_angle,
			float _motion_model_max_translation,
			float _max_particle_size,
			float _max_particle_size_kidnapped_robot,
			float _distanceAboveZ,
			float _rgd_resolution,
			int _max_number_considered_in_INNER_bucket,
			int _max_number_considered_in_OUTER_bucket,
			float _overlap_threshold,
			float _propability_threshold,
			float _rgd_2D_res);
	void setSolutionOffset(Eigen::Affine3f solutionOffset)
	{
		this->solutionOffset = solutionOffset;
	}
	Eigen::Affine3f getSolutionOffset()
	{
		return solutionOffset;
	}
	bool prediction(Eigen::Affine3f odometryIncrement);
	bool update();

	bool setCUDADevice(int _cudaDeveice);
	bool setGroundPointsFromMap(pcl::PointCloud<pcl::PointXYZ> pc);
	bool computeRGD();
	bool copyReferenceModelToGPU(pcl::PointCloud<Semantic::PointXYZL> &reference_model);
	bool copyCurrentScanToGPU(pcl::PointCloud<Semantic::PointXYZL> &current_scan);
	bool transformCurrentScan(Eigen::Affine3f matrix);
	void genParticlesKidnappedRobot();
	bool findClosestParticle(Eigen::Vector3f _reference_pose,
			Eigen::Vector3f &out_particle);
	Eigen::Affine3f getWinningParticle();

	void render();

	void setPose (geometry_msgs::PoseWithCovarianceStamped pose);
	geometry_msgs::PoseArray getPoseArray();
	nav_msgs::Odometry getOdom();

//

private:

	float randFloat()
	{
		return static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
	}
	float distance_above_Z;
	pcl::PointCloud<pcl::PointXYZ> ground_points_from_map;
	pcl::PointXYZ *d_ground_points_from_map;
	int number_of_points_ground_points_from_map;
	std::vector<particle_t> vparticles;
	float motion_model_max_angle;
	float motion_model_max_translation;
	int max_particle_size;
	int max_particle_size_kidnapped_robot;

	int cudaDevice;
	int number_of_threads;

	pcl::PointCloud<Semantic::PointXYZL> h_second_point_cloud;
	std::vector<int> h_nearest_neighbour_indexes;

	Semantic::PointXYZL *d_first_point_cloud;
	int number_of_points_first_point_cloud;
	Semantic::PointXYZL *d_second_point_cloud;
	Semantic::PointXYZL *d_second_point_cloudT;
	int *d_nearest_neighbour_indexes;
	int number_of_points_second_point_cloud;

	float bounding_box_extension;
	gridParameters rgd_params;
	float rgd_res;

	gridParameters rgd_params_2D;
	hashElement* d_hashTable_2D;
	bucket* d_buckets_2D;
	float rgd_2D_res;

	float *d_m;

	int max_number_considered_in_INNER_bucket;
	int max_number_considered_in_OUTER_bucket;
	float overlap_threshold;
	float propability_threshold;

	char *d_rgd;

	Eigen::Affine3f solutionOffset;
};



#endif
