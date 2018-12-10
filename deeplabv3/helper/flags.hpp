#include <gflags/gflags.h>
#include <iostream>

namespace helper{

DEFINE_string(
    image,
    "",
    "input image"
);

bool ValidateName(const char* flag, const std::string& value){
    if(value.empty()){
        std::cout << "Please set input image. --image <IMG_PATH>" << std::endl;
        return false;
    }
    return true;
}

}