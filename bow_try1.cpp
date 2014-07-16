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
#include "opencv2/opencv.hpp"

int main ( int argc, char *argv[] )
{
	
	//--- GLOBAL VARIABLES ---//
	//reading training and test set
	int database_size = 175;
	int no_classes = 3;
	int class_count[] = {53,57};
	int class_jump[] = {1060,1123};
	int no_im = 1000;
	int ccounter = 0;
	int imcounter = 0;
	std::string sno_im;
	std::ostringstream convert_int;
	cv::Mat img;
	std::string trainset_path = "training_sets/flavia_leaves/";
	std::string testset_path = "test_sets/flavia_leaves/";
	std::string stest;
	int no_tests = 19;
	//for extracting descriptors
	int minHess = 400;
	cv::SurfFeatureDetector surf_detector(minHess);
	cv::SurfDescriptorExtractor surf_extractor;
	cv::vector<cv::KeyPoint> surf_keypoints;
	cv::Mat surf_descriptors;
	cv::Mat local_descriptors;
	int max_nodescriptors = 0;
	//for building a dictionary
	cv::TermCriteria tc(CV_TERMCRIT_ITER,100,0.001);
	int cluster_attempts = 1;
	int dictionary_size = 1000; 
	cv::BOWKMeansTrainer bowTrainer(dictionary_size,tc,cluster_attempts, cv::KMEANS_PP_CENTERS);
	//for enconding images with BOW
	cv::Mat bowDescriptor;
	cv::Ptr<cv::DescriptorMatcher> dmatcher = cv::DescriptorMatcher::create("FlannBased");	
	cv::Ptr<cv::DescriptorExtractor> dextractor = cv::DescriptorExtractor::create("SURF");
	cv::BOWImgDescriptorExtractor bowDE(dextractor,dmatcher);
	//for classification
	cv::NormalBayesClassifier nb_classifier;
	cv::Mat training_data(0,dictionary_size,CV_32FC1);
	cv::Mat labels(0,1,CV_32FC1);
	cv::Mat eval_data(0,dictionary_size,CV_32FC1);
	cv::Mat results;
	//for measuring processing time
	clock_t t;	
	// --- ---//
	
	//--- DESCRIPTORS EXTRACTION ---//
	t = clock();
	for(int i = 0; i < database_size; i++)
	{

		if(ccounter < (no_classes-1) && imcounter == class_count[ccounter] )
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
		bowTrainer.add(local_descriptors);
		surf_descriptors.push_back(local_descriptors);
		surf_keypoints.clear();

		if(local_descriptors.rows > max_nodescriptors)
			max_nodescriptors = local_descriptors.rows;

		~local_descriptors;
		imcounter++;

	}

	std::cout << ccounter << " " << sno_im << std::endl;
	std::cout << surf_descriptors.size() << std::endl;
	std::cout << "Max no of descriptors: " << max_nodescriptors << std::endl;
	t = clock()-t;
	std::cout << "Descriptors processing time:" << std::endl;
	std::cout << t << " clicks " << ((float)t)/CLOCKS_PER_SEC << " seconds" << std::endl;
	//--- ---//

	 	
	//--- BUILD A DICTIONARY ---//
	t = clock();

	cv::Mat my_dictionary = bowTrainer.cluster();
	t = clock()-t;
	std::cout << "Dictionary processing time:" << std::endl;
	std::cout << t << " clicks " << ((float)t)/CLOCKS_PER_SEC << " seconds" << std::endl;
	//--- ---//

	//--- Naive Bayes for classification ---//

	no_im = 1000;
	ccounter = 0;
	imcounter = 0;
	convert_int.str("");
	convert_int.flush();
	surf_keypoints.clear();
	//set the dictionary
	bowDE.setVocabulary(my_dictionary);	

	for(int i = 0; i < database_size; i++)
	{
		if(ccounter < (no_classes-1) && imcounter == class_count[ccounter] )
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
	//	std::cout << sno_im << std::endl;
		img = cv::imread(sno_im,CV_LOAD_IMAGE_COLOR);	
		surf_detector.detect(img,surf_keypoints);
		bowDE.compute(img,surf_keypoints,bowDescriptor);
	//	std::cout << bowDescriptor.size() << " ppp" << std::endl;
		training_data.push_back(bowDescriptor);
		labels.push_back(float(ccounter+1));
		surf_keypoints.clear();
		~bowDescriptor;
		imcounter++;

	}
	
	std::cout << "Training classifier..." << std::endl;
	std::cout << training_data.size() << " * " << labels.size() << std::endl;
	if(training_data.type() == CV_32FC1)
	{std::cout << "yes1" << std::endl;}
	if(labels.type() == CV_32FC1)
	{std::cout << "yes2" << std::endl;}

	t = clock();
	nb_classifier.train(training_data,labels,cv::Mat(),cv::Mat(),false);
	t = clock() - t;
	std::cout << " Training processing time:" << std::endl;
	std::cout << t << " clicks " << ((float)t)/CLOCKS_PER_SEC << " seconds" << std::endl;

	//--- ---//

	//--- Evaluation images BOW encoding ---//

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
		eval_data.push_back(bowDescriptor);
		surf_keypoints.clear();
		~bowDescriptor;
	}

	t = clock()-t;
	std::cout << " Image BOW encoding time:" << std::endl;
	std::cout << t << " clicks " << ((float)t)/CLOCKS_PER_SEC << " seconds" << std::endl;

	//--- ---//	

	//--- Evaluation of test set ---//
	std::cout << "Evaluating classifier..." << std::endl;
	t = clock();
	nb_classifier.predict(eval_data,&results);	
	t = clock()-t;
	std::cout << " Classifier evaluation time:" << std::endl;
	std::cout << t << " clicks " << ((float)t)/CLOCKS_PER_SEC << " seconds" << std::endl;

	std::cout << "Classifier Results" << std::endl;
	std::cout << results << std::endl;

	return 0;
}				/* ----------  end of function main  ---------- */

