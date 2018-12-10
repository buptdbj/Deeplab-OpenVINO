#include <inference_engine.hpp>
#include <algorithm>
#include <string>
#include <vector>
#include <iostream>
#include <map>
#include <fstream>
#include <gflags/gflags.h>

#include <ext_list.hpp>
#include <format_reader_ptr.h>
#include <samples/common.hpp>
#include <opencv_wraper.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "tools.hpp"

using namespace InferenceEngine;
using namespace std;

DEFINE_string(
    image,
    "",
    "input image"
);

int main(int argc, char* argv[]){
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    if(FLAGS_image.empty()){
        throw std::logic_error("Parameter -image is not set");
    }
    auto version = GetInferenceEngineVersion();
    cout << "InferenceEngine Version: " << version->apiVersion.major << "." << version->apiVersion.minor << endl;
    cout << "build: " << version->buildNumber << endl;
    // 1. Load a Plugin
    vector<string> pluginDirs {"/home/sfy/intel/computer_vision_sdk/deployment_tools/inference_engine/lib/ubuntu_16.04/intel64"};
    InferenceEnginePluginPtr engine_ptr = PluginDispatcher(pluginDirs).getSuitablePlugin(TargetDevice::eCPU);
    InferencePlugin plugin(engine_ptr);
    cout << "Plugin Version: " << plugin.GetVersion()->apiVersion.major << "." << plugin.GetVersion()->apiVersion.minor << endl;
    cout << "build: " << plugin.GetVersion()->buildNumber << endl;
    plugin.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>());

    // 2. Read the Model Intermediate Representation (IR)
    CNNNetReader network_reader;
    network_reader.ReadNetwork("/home/sfy/ws/ir/deeplab/frozen_inference_graph.xml");
    network_reader.ReadWeights("/home/sfy/ws/ir/deeplab/frozen_inference_graph.bin");

    // 3. Configure Input and Output
    CNNNetwork network = network_reader.getNetwork();
    network.setBatchSize(1);

    /** Taking information about all topology inputs **/
    InferenceEngine::InputsDataMap input_info(network.getInputsInfo());
    /** Taking information about a`ll topology outputs **/
    InferenceEngine::OutputsDataMap output_info(network.getOutputsInfo());

    for(
        map<string, InputInfo::Ptr>::iterator it = input_info.begin(); 
        it != input_info.end();
        it ++){
        it->second->setPrecision(Precision::U8);
        cout << "Input: " << it->first << endl
            << "\tPrecision: " << it->second->getPrecision() << endl;
        // it->second->setLayout(Layout::NHWC);
        cout << "\tDim: [ ";
        for(auto x: it->second->getDims()){
            cout << x << " ";
        }
        cout << "]" << endl;
    }

    for(
        map<std::string, DataPtr>::iterator it = output_info.begin();
        it != output_info.end();
        it ++){
        it->second->setPrecision(Precision::FP32);
        cout << "Output: " << it->first << endl
            << "\tPrecision: " << it->second->getPrecision() << endl;
        cout << "\tDim: [ ";
        for(auto x: it->second->dims){
            cout << x << " ";
        }
        cout << "]" << endl;
    }

    // 4. Load the Model
    ExecutableNetwork executable_network = plugin.LoadNetwork(network, {});
    // 5. Create Infer Request
    InferRequest infer_request = executable_network.CreateInferRequest();

    // 6. Prepare Input
    /** Collect images data ptrs **/
    FormatReader::ReaderPtr reader(FLAGS_image.c_str());
    if (reader.get() == nullptr) {
        cout << "Image: " << FLAGS_image << " cannot be read!" << endl;
        return -1;
    }
    
    string input_name = (*input_info.begin()).first;
    Blob::Ptr input = infer_request.GetBlob(input_name);
    size_t num_channels = input->getTensorDesc().getDims()[1];
    size_t image_size = input->getTensorDesc().getDims()[3] * input->getTensorDesc().getDims()[2];


    /** Getting image data **/
    std::shared_ptr<unsigned char> imageData(reader->getData(input->getTensorDesc().getDims()[3],
                                                             input->getTensorDesc().getDims()[2]));

    /** Iterating over all input blobs **/
    cout << "Prepare Input" << endl;
    /** Getting input blob **/
    /** Fill input tensor with planes. First b channel, then g and r channels **/
    auto data = input->buffer().as<PrecisionTrait<Precision::U8>::value_type*>();
    /** Setting batch size using image count **/

    /** Iterate over all input images **/
    /** Iterate over all pixel in image (r,g,b) **/

    for (size_t ch = 0; ch < num_channels; ch++) {
        /** Iterate over all channels **/
        for (size_t pid = 0; pid < image_size; pid++) {
            /** [images stride + channels stride + pixel id ] all in bytes **/
            data[0 * image_size * num_channels + ch * image_size + pid] = imageData.get()[pid*num_channels + ch];
        }
    }

    // 7. Perform Inference
    infer_request.Infer();

    // 8. Process Output
    cout << "Processing output blobs" << endl;

    string output_name = (*output_info.begin()).first;
    const Blob::Ptr output_blob = infer_request.GetBlob(output_name);
    const float* output_data = output_blob->buffer().as<float*>();

    size_t N = 1;
    size_t C = output_blob->getTensorDesc().getDims().at(0);
    size_t H = output_blob->getTensorDesc().getDims().at(1);
    size_t W = output_blob->getTensorDesc().getDims().at(2);

    size_t image_stride = W*H;

    cv::Mat Seg (H, W, 0);
    for (int i = 0; i < image_size; i++){
        Seg.data[i] = output_data[i];
    }
    cv::Mat croppedSeg = Seg(cv::Rect(0, 0, reader->resized_w, reader->resized_h));
    cv::Mat resizedCroppedSeg (reader->width(), reader->height(), 0);
    cv::resize(croppedSeg, resizedCroppedSeg, cv::Size(reader->width(), reader->height()));
    cv::imshow("seg", Seg);
    cv::imshow("resizedCroppedSeg", resizedCroppedSeg);

    /** Dump resulting image **/
    std::string fileName = "out.bmp";
    std::ofstream outFile(fileName, std::ofstream::binary);
    if (!outFile.is_open()) {
        throw std::logic_error("Can't open file : " + fileName);
    }

    overlayOutput(resizedCroppedSeg, reader->img, 21);

    cv::waitKey(0);
}
