#include <opencv2/opencv.hpp>
#include "RegionSupportingTree.h"
#include "LayerMerging.h"
#include "LayerVectorizing.h"
#include "Region.h"
#include <algorithm>
# include<ctime>
using namespace std;

// #include <filesystem>

int main(int argc, char** argv) {

	string main_dir = "../../Data/";
    // string cases[] = { "", "1-Apple", "2-Apple-SAM", "3-Orange", "4-Orange-SAM", "5-Sushi", "6-Sushi-SAM" };
	// int id = 5;
	assert(argc == 2);

	// for (int i = id; i < id+1; i++) {
		// cout << "\n\nCase " << i << " : " << cases[i] <<"======================\n\n" ;
		// string data_dir = main_dir + cases[i];
		cout << "\n\nCase " << argv[1] <<"======================\n\n" ;
		string data_dir = main_dir + argv[1];
		string input_img_path = data_dir + "/input.png";
		string input_region_img_path = data_dir + "/region.png";
		string input_region_info_path = data_dir + "/region_info.txt";
		string ouput_vectorize_path = data_dir + "/results/";

		//used to generate vector graph
		string output_json_path = data_dir + "/results/for_vectorize/";
		string output_layer_mask_path = data_dir + "/results/for_vectorize/";

		//0. get input==============================================================
		cout << "0. read original image, region image and other region info...\n" << endl;
		ImageObj ori_img(input_img_path);
		RegionInfo RegInfo(input_region_img_path, input_region_info_path);

		//1. generate region order trees============================================
		cout << "1. start to generate region supporting trees...\n" << endl;
		clock_t t0 = clock();
		RegionSupportingTree Rst(ori_img, RegInfo.regions, RegInfo.xjunction, RegInfo.possible_bottom_rids);
		Rst.BuildAdjacentRegionGraph();
		Rst.GetValidRegionSupportingTrees();
		vector<Tree> ValidRegSupportTrees = Rst.m_valid_region_support_trees;
		// if (ValidRegSupportTrees.empty()) continue;
		if (ValidRegSupportTrees.empty()) return -1;
		clock_t t1 = clock();

		//2. layer merging==========================================================
		cout << "2. start to merge layers...\n" << endl;
		int tree_cnt = min(100, (int)ValidRegSupportTrees.size());
		vector<LayerMerging> LMs(tree_cnt);
#pragma omp parallel for
		for (int ind = 0; ind < tree_cnt; ind++) {
			Tree tree = ValidRegSupportTrees[ind];
			LMs[ind] = LayerMerging(Rst.m_regions, tree);
			LMs[ind].DetermineLayerRange();
		}

		//2.1 deduplicate layer configurations
		for (int i = 0; i < LMs.size() - 1; i++) {
			for (int j = LMs.size() - 1; j > i; j--)
				if (LMs[i].LayerConfigurationEquals(LMs[j]))
					LMs.erase(LMs.begin() + j);
		}
		cout << "after deduplicate, tree cnt:" << LMs.size() << endl;
		clock_t t2 = clock();

		//3. layer parameter optimization====================================================
		cout << "3. start to estimate layer parameters...\n" << endl;
		vector<LayerVectorizing> LVs(LMs.size());
		cout << "layer cnt:" << LMs.size() << endl;
#pragma omp parallel for
		for (int ind = 0; ind < LVs.size(); ind++) {
			LVs[ind] = LayerVectorizing(Rst.m_regions,  &ori_img, LMs[ind].GetLayerObject());
			LMs[ind].Release();
			LVs[ind].CalculateLayerObjectParamsWithGlobalOptimization();
			LVs[ind].CalculateTotalLoss();
			cout << "config " << ind << " has been decomposed!" << endl;
		}
		clock_t t3 = clock();

		//4. output top 5 results============================================================
		sort(LVs.begin(), LVs.end());
		cout << endl << "4. start to output layer and reconstruted image...\n" << endl;
#pragma omp parallel for
		for (int ind = 0; ind < max(min((int)LVs.size(), 5),1); ind++) {
			LVs[ind].GenerateResultingLayers();
			LVs[ind].SaveReconstructedImageAndLayers(ouput_vectorize_path + to_string(ind));
			LVs[ind].OutputJsonForPresentation(output_json_path + to_string(ind) + "/param.json");
			LVs[ind].OutputLayerMask(output_layer_mask_path + to_string(ind) + "/");
		}
	// }
	return 0;
}