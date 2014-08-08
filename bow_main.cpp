/* =============================================================*/
/* ---                 BOW MAIN - SOURCE FILE                ---*/
/* FILENAME: bow_main.cpp 
 *
 * DESCRIPTION: source file to run the full implementation of the
 * Bag of Words algorithm. Naive Bayes is used for classification.
 *
 * VERSION: 1.0
 *
 * CREATED: 07/18/2014
 *
 * COMPILER: g++
 *
 * AUTHOR: ARTURO GOMEZ CHAVEZ
 * ALIAS: BOSSLEGEND33
 * 
 * ============================================================ */

//c++ libraries
#include <iostream>
#include <vector>
#include <string>
#include <time.h>
#include <fstream>
#include <iomanip>
#include <stdlib.h>
#include "boost/filesystem.hpp"
//opencv libraries
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/ml/ml.hpp"
//Defines
#define TRAIN_PATH "training_sets/logistics_b"
#define TEST_PATH "test_sets/logistics_b"
#define DESCRIP_MAT_NAME "descrip_train_mat.yml"
//namespaces
namespace fs = boost::filesystem;

//--- GLOBAL VARIABLES ---//
//for detecting features
int minHess = 400;
cv::SurfFeatureDetector feature_detector(minHess);
//for enconding images with BOW
cv::Mat bowDescriptor;
cv::Ptr<cv::DescriptorMatcher> dmatcher = cv::DescriptorMatcher::create("FlannBased");	
cv::Ptr<cv::DescriptorExtractor> dextractor = cv::DescriptorExtractor::create("SURF");
cv::BOWImgDescriptorExtractor bowDE(dextractor,dmatcher);
//--- ---/

//codification of the images with the BOW vocabulary
void bow_encode(const fs::path& basepath, cv::Mat& descriptors, cv::Mat& labels)
{
	int no_files = 0;
	std::string class_name = basepath.string();
	class_name.erase(class_name.begin(),class_name.end()-2);

	for(fs::directory_iterator iter(basepath), end; iter != end; ++iter)
	{
		fs::directory_entry entry = *iter;
		if(fs::is_directory(entry.path()))
		{
			std::cout << "Processing directory: " << entry.path().string() << std::endl;
			bow_encode(entry.path(),descriptors,labels);
		}
		else
		{
			fs::path entryPath = entry.path();
			if(entryPath.extension()==".jpg")
			{
				//std::cout << "Processing file: " << entry.path().string() << std::endl;
				no_files++;
				cv::Mat img = cv::imread(entryPath.string(),CV_LOAD_IMAGE_COLOR);
				if(!img.empty())
				{
					std::vector<cv::KeyPoint> feature_keypoints;
					feature_detector.detect(img,feature_keypoints);
					if(feature_keypoints.empty())
					{
						std::cerr << "Could not find points in image: " << entryPath.string();
						std::cerr << std::endl;
					}
					else
					{

						cv::Mat bowDescriptor;
						bowDE.compute(img,feature_keypoints,bowDescriptor);
						descriptors.push_back(bowDescriptor);
						//std::cout << class_name.c_str() << std::endl;
						labels.push_back( float( atoi(class_name.c_str()) ) );
						feature_keypoints.clear();
						//std::cout << local_descriptors.rows << std::endl;	
						~bowDescriptor;

					}
				}
				else
				{
					std::cerr << "Could not read image: " << entryPath.string() << std::endl;
				}
												
			}												
		}							
		
	}		
	std::cout << "No files: " << no_files << std::endl;

}

void test_ranforest(CvRTrees* forest, cv::Mat& samples, cv::Mat& ground_truth, cv::Mat& results)
{
	//std::vector<double> results;	
	double pred_result = 0.;	
	int isPredCorrect = 0;
	int mis_classif = 0;

	for(int i = 0; i < samples.rows; i++)
	{
		pred_result = forest->predict(samples.row(i));
		results.push_back(pred_result);
		isPredCorrect = fabs(pred_result-ground_truth.at<float>(i)) >= FLT_EPSILON;
		
		if(isPredCorrect)
		{ mis_classif++;}
		
	}

	double error = (double)mis_classif/samples.rows;
	std::cout << "Wrong class: " << mis_classif << " No cases: " << samples.rows;
	std::cout << std::endl;
	std::cout << "Error %: " << error << std::endl;	



}


//Main function
int main ( int argc, char *argv[] )
{
	
	//---  VARIABLES ---//
	//saving database descriptors
	cv::Mat training_descriptors;
	//for measuring processing time
	clock_t t;	
	// --- ---//
	std::cout << std::endl;
	std::cout << "+++ BOW FOR DATA SET +++" << std::endl;
	std::cout << TRAIN_PATH << std::endl;

	//--- DESCRIPTORS EXTRACTION ---//
	std::string train_path = TRAIN_PATH;
	//read descriptors from file
	std::cout << "*** TRAIN DESCRIPTORS INFO ***" << std::endl;
	cv::FileStorage fstore_descrip(DESCRIP_MAT_NAME, cv::FileStorage::READ);
	std::cout << "No Documents: " << (int)fstore_descrip["noDocuments"] << std::endl;
	std::cout << "No Classes: " << (int)fstore_descrip["noClasses"] << std::endl;
	std::cout << "No total descriptors: " << (int)fstore_descrip["totalDescriptors"] << std::endl;
	std::cout << "No max desrcriptors in an image: " << (int)fstore_descrip["maxDescriptors"] << std::endl;
	std::cout << "Descriptors processing time: " << (float)fstore_descrip["procTime"] << std::endl;
	std::cout << std::endl;
	fstore_descrip["matDescriptors"] >> training_descriptors;
	fstore_descrip.release();
	

	//--- BUILD A DICTIONARY ---//
	cv::TermCriteria tc(CV_TERMCRIT_ITER,100,0.001);
	int cluster_attempts = 1;
	int dictionary_size = atoi(argv[1]);
	std::cout << "*** BOW DICTIONARY INFO ***" << std::endl;
	std::cout << "Dictionary size: " << dictionary_size << std::endl; 
	cv::BOWKMeansTrainer bowTrainer(dictionary_size,tc,cluster_attempts, cv::KMEANS_PP_CENTERS);

	bowTrainer.add(training_descriptors);
	t = clock();
	cv::Mat my_dictionary = bowTrainer.cluster();
	t = clock()-t;
	std::cout << "Dictionary processing time:" << std::endl;
	std::cout << t << " clicks " << ((float)t)/CLOCKS_PER_SEC << " seconds" << std::endl;
	std::cout << std::endl;
	//--- ---//

	//--- BOW CODIFICATION OF TRAINING SET ---//
	//set the dictionary for the bow decriptor extractor
	cv::Mat training_data(0,dictionary_size,CV_32FC1);
	cv::Mat labels(0,1,CV_32FC1);
	bowDE.setVocabulary(my_dictionary);
	bow_encode(fs::path(train_path),training_data,labels);

	std::cout << "Training size: " << training_data.rows << "x" << training_data.cols << std::endl;
	std::cout << "Labels size: " << labels.rows << "x" << labels.cols << std::endl;
	 
	//RANDOM FOREST CLASSIFICATION ---//
	CvERTrees* eforest;
	CvRTrees* forest;
	cv::Mat var_type(training_data.cols+1,1, CV_8U, cv::Scalar(CV_VAR_ORDERED));
	var_type.at<unsigned char>(training_data.cols,0) = CV_VAR_CATEGORICAL;
	//cv::Mat mask_vt(training_data.cols+1,1, CV_8U, cv::Scalar(0));
	//mask_vt.at<unsigned char>(training_data.cols) = 1;
	//var_type.setTo(CV_VAR_CATEGORICAL, mask_vt);

	float priors[] = {1., 1.};
	std::cout << "Training random forest..." << std::endl;
	forest = new CvRTrees;
	std::cout << "No. samples: " << training_data.rows << std::endl;
	forest = new CvRTrees;
	t = clock();
	forest->train(	training_data,	//cv::Mat containing samples and their attribute values
			CV_ROW_SAMPLE,	//defines if there is a sample ine very row or col
			labels,		//vector containing the responses of every sample
			cv::Mat(),	//vector to indicate which attributes to consider for the training (0-skip)
			cv::Mat(),	//vector to indicate which samples to consider for the training (0-skip)
			var_type,	//matrix that states if each feature is ordered or categorical
			cv::Mat(),	//matrix used to indicate missing values with a 1
			CvRTParams(	150, 	//max depth of the tree
					1,	//min number of samples in a node to make a split
					0,	//regression acuracy, N/A for categorical, termination criteria for regression
					true,  //compute surrogate splits
					100, 	//max number of categories
					0,//priors,	//array of priors (weights)
					true,	//calculate var importance
					0,	//active vars, number of variables used to build each tree node
					300,	//max number of trees in the forest
					0.01,  //sufficient accuracy (OOB error)
					CV_TERMCRIT_ITER //termination criteria, by reaching max number of trees and/or accuracy
			));

	std::cout << "Ready" << std::endl;
	t = clock()-t;
	std::cout << " Classifier training time:" << std::endl;
	std::cout << t << " clicks " << ((float)t)/CLOCKS_PER_SEC << " seconds" << std::endl;
	std::cout << "No trees: " << forest->get_tree_count() << std::endl;
	std::cout << "Calculating training error..." << std::endl;
	cv::Mat r;
	test_ranforest(forest,training_data, labels, r);
	
	//RT for prediction
	cv::Mat ground_truth(0,1,CV_32FC1);
	cv::Mat eval_data(0,dictionary_size,CV_32FC1);
	cv::Mat results;
	std::string test_path = TEST_PATH;

	std::cout << "*** CLASSIFIER EVALUATION ***" << std::endl;
	bow_encode(fs::path(test_path),eval_data,ground_truth);
	t = clock();
	test_ranforest(forest, eval_data, ground_truth, results);
	t = clock()-t;
	std::cout << " Classifier evaluation time:" << std::endl;
	std::cout << t << " clicks " << ((float)t)/CLOCKS_PER_SEC << " seconds" << std::endl;
	std::cout << "Classifier Results" << std::endl;
	std::cout << results << std::endl << std::endl;

	/*  */

	~ground_truth;
	~eval_data;
	
	//--- NAIVE BAYES FOR CLASSIFICATION ---//
	cv::NormalBayesClassifier nb_classifier;

	std::cout << "*** CLASSIFIER TRAINING ***" << std::endl;
	// +++ for debugging - can  be commented +++//
	std::cout << training_data.size() << " * " << labels.size() << std::endl;
	if(training_data.type() == CV_32FC1)
	{std::cout << "training data matrix accepted" << std::endl;}
	if(labels.type() == CV_32FC1)
	{std::cout << "labels matrix accepted" << std::endl;}
	// +++ +++ //

	t = clock();
	nb_classifier.train(training_data,labels,cv::Mat(),cv::Mat(),false);
	t = clock() - t;
	//nb_classifier.save("nbModel_logistics_b.yml","nbModel_logistics_b");
	std::cout << " Training processing time:" << std::endl;
	std::cout << t << " clicks " << ((float)t)/CLOCKS_PER_SEC << " seconds" << std::endl;
	std::cout << std::endl;
	
	//if you already have a classifier uncomment the next line and comment the all the
	//Classifier Training section
	//nb_classifier.load("nbModel_flavia_leaves_b.yml","nbModel_flavia_leaves_b");
	//std::cout << "Classfier model loaded"<< std::endl;

	//--- ---//

	//--- BOW ENCODING OF TEST SET AND EVALUATION ---//
	cv::Mat ground_truth2(0,1,CV_32FC1);
	cv::Mat eval_data2(0,dictionary_size,CV_32FC1);
	cv::Mat results2;
	double accuRate = 0.;

	std::cout << "*** CLASSIFIER EVALUATION ***" << std::endl;
	bow_encode(fs::path(test_path),eval_data2,ground_truth2);
	t = clock();
	nb_classifier.predict(eval_data2,&results2);	
	t = clock()-t;
	std::cout << " Classifier evaluation time:" << std::endl;
	std::cout << t << " clicks " << ((float)t)/CLOCKS_PER_SEC << " seconds" << std::endl;

	accuRate = 1. -( (double) cv::countNonZero(ground_truth2 - results2) / eval_data2.rows);
	std::cout << "Accuracy rate: " << accuRate << std::endl;
	std::cout << "Classifier Results" << std::endl;
	std::cout << results2 << std::endl << std::endl;
	/*  */	
	

	return 0;
}				/* ----------  end of function main  ---------- */


