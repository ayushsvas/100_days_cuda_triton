#include <torch/extension.h>
#include <iostream.h>


int main(){
    std::cout << "LibTorch Version" << TORCH_VERSION << std::endl;
    try {
        torch::Tensor t = torch::zeros({2, 2}, torch::TensorOptions().dtype(torch::kComplexDouble));
        std::cout << "Complex double tensor created successfully." << std::endl;
    } catch (const c10::Error& e) {
        std::cerr << "Error creating complex double tensor: " << e.what() << std::endl;
    }

    return 0;
}