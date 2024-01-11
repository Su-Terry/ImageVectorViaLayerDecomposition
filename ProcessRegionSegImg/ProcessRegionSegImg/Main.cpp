
#include "Utility.h"
#include <iostream>
// #include <direct.h>
// #include <io.h>
#include "RegionInfo.h"

int main(int argc, char** argv) { 

    string main_dir = "../../Data/";
    // string cases[] = { "", "1-Apple", "2-Apple-SAM", "3-Orange", "4-Orange-SAM", "5-Sushi", "6-Sushi-SAM" };
    // int id = 5;
    assert(argc == 2);

    // for (int i = id; i < id+1; i++) {
        // cout << "\n\nCase " << i << " : " << cases[i] << endl << endl;
        // string data_dir = main_dir + cases[i];
        cout << "\n\nCase " << argv[1] << endl << endl;
        string data_dir = main_dir + argv[1];
        string input_img_path = data_dir + "/input.png";
        string input_seg_path = data_dir + "/seg.png";
        string input_mask = data_dir + "/mask.png";
        string output_region_path = data_dir + "/region.png";
        string output_param_path = data_dir + "/region_info.txt";
        string output_region_ind_path = data_dir + "/region_index.png";

        RegionInfo Ri(input_img_path, input_seg_path,input_mask);
        Ri.OutputRegionInfo_s1(output_region_path, output_param_path, output_region_ind_path);
        Ri.GetAdjacencyInfo(output_param_path);
        // Ri.GetXjunctionInfo(output_param_path);
        Ri.OutputRegionInfo_s2(output_param_path);
    // }
    return 0;
}