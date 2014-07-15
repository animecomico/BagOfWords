/*
 * =====================================================================================
 *
 *       Filename:  bow_try1.cpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  07/14/2014 02:09:13 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */

//c++ libraries
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <time.h>
//opencv libraries
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/highgui/highgui.hpp"


void draw_circles(cv::Mat im, std::vector<cv::KeyPoint> keyp, const cv::Scalar &colour)
{
	std::vector<cv::Point2f> pts;
	cv::KeyPoint::convert(keyp,pts);

	std::vector<cv::Point2f>::const_iterator it = pts.begin();
	while(it != pts.end())
	{
		cv::circle(im,*it,3,colour,2);
		++it;
	}
}

int main ( int argc, char *argv[] )
{
	/* 
	cv::Mat img1 = cv::imread("training_sets/1006.jpg", CV_LOAD_IMAGE_COLOR);

	//--- COMPUTING DESCRIPTORS ---//
	//find keypoints
	cv::SurfFeatureDetector surf_detector(400);
	std::vector<cv::KeyPoint> surf_keypoints;
	surf_detector.detect(img1,surf_keypoints);
	//compute descriptors
	cv::SurfDescriptorExtractor surf_extractor;
	cv::Mat surf_descriptors;
	surf_extractor.compute(img1,surf_keypoints,surf_descriptors);
	// --- --- //

	draw_circles(img1, surf_keypoints, cv::Scalar(255,0,0));

	cv::namedWindow("Test", CV_WINDOW_NORMAL);
	cv::imshow("Test",img1);

	cv::waitKey(0);
	cv::waitKey(0);	
	*/


	//--- DESCRIPTORS EXTRACTION ---//
	int database_size = 175;
	int class_count[] = {53,57};
	int class_jump[] = {1060,1123};
	int no_im = 1000;
	int ccounter = 0;
	int imcounter = 0;
	std::string sno_im;
	std::ostringstream convert_int;
	cv::Mat img;
	std::string trainset_path = "training_sets/flavia_leaves/";
	int minHess = 400;
	cv::SurfFeatureDetector surf_detector(minHess);
	cv::SurfDescriptorExtractor surf_extractor;
	cv::vector<cv::KeyPoint> surf_keypoints;
	cv::Mat surf_descriptors;
	cv::Mat local_descriptors;

	clock_t t;
	t = clock();

	for(int i = 0; i < database_size; i++)
	{

		if(imcounter == class_count[ccounter] )
		{
			no_im = class_jump[ccounter];
			ccounter++;
			imcounter = 0;
		}
		else
		{++no_im;}
		convert_int << trainset_path << no_im << ".jpg";
		sno_im = convert_int.str();
		convert_int.str("");
		convert_int.flush();

		img = cv::imread(sno_im,CV_LOAD_IMAGE_COLOR);
		surf_detector.detect(img,surf_keypoints);
		surf_extractor.compute(img,surf_keypoints,local_descriptors);
		surf_descriptors.push_back(local_descriptors);			
		surf_keypoints.clear();
		//std::cout << local_descriptors.size() << std::endl;
		~local_descriptors;
		imcounter++;


	}

	//std::cout << surf_descriptors.size() << std::endl;

	t = clock()-t;
	std::cout << "Descriptors processing time:" << std::endl;
	std::cout << t << " clicks " << ((float)t)/CLOCKS_PER_SEC << " seconds" << std::endl;
	//--- ---//

	 	
	//--- BUILD A DICTIONARY ---//
	t = clock();
	cv::TermCriteria tc(CV_TERMCRIT_ITER,100,0.001);
	int cluster_attempts = 1;
	int dictionary_size = 1000; 

	cv::BOWKMeansTrainer bowTrainer(dictionary_size,tc,cluster_attempts, cv::KMEANS_PP_CENTERS);
	cv::Mat my_dictionary = bowTrainer.cluster(surf_descriptors);
	t = clock()-t;
	std::cout << "Dictionary processing time:" << std::endl;
	std::cout << t << " clicks " << ((float)t)/CLOCKS_PER_SEC << " seconds" << std::endl;
	//--- ---//

	//--- Evaluation images BOW encoding ---//
	cv::Ptr<cv::DescriptorMatcher> dmatcher = cv::DescriptorMatcher::create("FlannBased");	
	cv::Ptr<cv::DescriptorExtractor> dextractor = cv::DescriptorExtractor::create("SURF");
	//set the dictionary
	cv::BOWImgDescriptorExtractor bowDE(dextractor,dmatcher);
	bowDE.setVocabulary(my_dictionary);	

	std::string testset_path = "test_sets/flavia_leaves/";
	std::string stest;
	int no_tests = 19;
	cv::Mat bowDescriptor;

	t = clock();

	for(int i = 0; i < no_tests; i++)
	{
		convert_int.str("");
		convert_int.flush();
		convert_int << testset_path << (i+1) << ".jpg";
		stest = convert_int.str();

		img = cv::imread(stest,CV_LOAD_IMAGE_COLOR);
		surf_detector.detect(img,surf_keypoints);
		bowDE.compute(img,surf_keypoints,bowDescriptor);
		surf_keypoints.clear();
		~bowDescriptor;
	}

	t = clock()-t;
	std::cout << " Image BOW encoding time:" << std::endl;
	std::cout << t << " clicks " << ((float)t)/CLOCKS_PER_SEC << " seconds" << std::endl;


	return 0;
}				/* ----------  end of function main  ---------- */

