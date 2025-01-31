#include <shapeDescriptor/cpu/types/Mesh.h>
#include <shapeDescriptor/gpu/spinImageGenerator.cuh>
#include <shapeDescriptor/gpu/radialIntersectionCountImageGenerator.cuh>
#include <shapeDescriptor/gpu/quickIntersectionCountImageGenerator.cuh>
#include <shapeDescriptor/utilities/read/OBJLoader.h>
#include <shapeDescriptor/utilities/dump/descriptorImages.h>
#include <shapeDescriptor/utilities/CUDAContextCreator.h>
#include <shapeDescriptor/utilities/free/mesh.h>

#include <arrrgh.hpp>
#include <shapeDescriptor/utilities/copy/mesh.h>
#include <shapeDescriptor/utilities/kernels/gpuMeshSampler.cuh>
#include <shapeDescriptor/utilities/copy/array.h>
#include <shapeDescriptor/utilities/read/MeshLoader.h>
#include <shapeDescriptor/utilities/free/array.h>
#include <shapeDescriptor/utilities/CUDAAvailability.h>
#include <shapeDescriptor/cpu/radialIntersectionCountImageGenerator.h>
#include <shapeDescriptor/utilities/spinOriginsGenerator.h>
#include <shapeDescriptor/cpu/quickIntersectionCountImageGenerator.h>
#include <shapeDescriptor/utilities/meshSampler.h>

int main(int argc, const char** argv) {
    const std::string defaultExecutionDevice = ShapeDescriptor::isCUDASupportAvailable() ? "gpu" : "cpu";

    arrrgh::parser parser("imagerenderer", "Generate RICI or spin images from an input object and dump them into a PNG file");
    const auto& inputFile = parser.add<std::string>(
            "input", "The location of the input OBJ model file.", '\0', arrrgh::Required, "");
    const auto& showHelp = parser.add<bool>(
            "help", "Show this help message.", 'h', arrrgh::Optional, false);
    const auto& generationMode = parser.add<std::string>(
            "image-type", "Which image type to generate. Can either be 'si', 'rici', or 'quicci'.", '\0', arrrgh::Optional, "rici");
    const auto& forceGPU = parser.add<int>(
            "force-gpu", "Force using the GPU with the given ID", 'b', arrrgh::Optional, -1);
    const auto& generationDevice = parser.add<std::string>(
            "device", "Determines whether to compute the images on the CPU or GPU, by specifying its value as 'cpu' or 'gpu', respectively.", '\0', arrrgh::Optional, defaultExecutionDevice);
    const auto& spinImageWidth = parser.add<float>(
            "support-radius", "The size of the spin image plane in 3D object space", '\0', arrrgh::Optional, 1.0f);
    const auto& imageLimit = parser.add<int>(
            "image-limit", "The maximum number of images to generate (in order to limit image size)", '\0', arrrgh::Optional, -1);
    const auto& enableLogarithmicImage = parser.add<bool>(
            "logarithmic-image", "Apply a logarithmic filter on the image to better show colour variation.", 'l', arrrgh::Optional, false);
    const auto& supportAngle = parser.add<float>(
            "spin-image-support-angle", "The support angle to use for spin image generation", '\0', arrrgh::Optional, 90.0f);
    const auto& spinImageSampleCount = parser.add<int>(
            "spin-image-sample-count", "The number of uniformly sampled points to use for spin image generation", '\0', arrrgh::Optional, 1000000);
    const auto& imagesPerRow = parser.add<int>(
            "images-per-row", "The number of images the output image should contain per row", '\0', arrrgh::Optional, 50);
    const auto& outputFile = parser.add<std::string>(
            "output", "The location of the PNG file to write to", '\0', arrrgh::Optional, "out.png");

    try
    {
        parser.parse(argc, argv);
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error parsing arguments: " << e.what() << std::endl;
        parser.show_usage(std::cerr);
        exit(1);
    }

    // Show help if desired
    if(showHelp.value())
    {
        return 0;
    }

    if(forceGPU.value() != -1) {
        std::cout << "Forcing GPU " << forceGPU.value() << std::endl;
        ShapeDescriptor::utilities::createCUDAContext(forceGPU.value());
    }

    if(!ShapeDescriptor::isCUDASupportAvailable() && generationDevice.value() == "gpu") {
        throw std::runtime_error("Image generation on the GPU was requested, but libShapeDescriptor was compiled GPU kernels disabled.");
    }

    std::cout << "Loading mesh file.." << std::endl;
    ShapeDescriptor::cpu::Mesh mesh = ShapeDescriptor::utilities::loadMesh(inputFile.value(), true);
    std::cout << "    Object has " << mesh.vertexCount << " vertices" << std::endl;

    ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> spinOrigins = ShapeDescriptor::utilities::generateUniqueSpinOriginBuffer(mesh);
    std::cout << "    Found " << spinOrigins.length << " unique vertices" << std::endl;

    // Limit image count being generated depending on command line parameter
    if(imageLimit.value() != -1) {
        spinOrigins.length = std::min<size_t>(spinOrigins.length, imageLimit.value());
    }

    ShapeDescriptor::gpu::Mesh deviceMesh;
    ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> deviceSpinOrigins;

    if(ShapeDescriptor::isCUDASupportAvailable() && generationDevice.value() == "gpu") {
        deviceMesh = ShapeDescriptor::copy::hostMeshToDevice(mesh);

        ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> tempOrigins = ShapeDescriptor::copy::hostArrayToDevice(spinOrigins);
        deviceSpinOrigins = {tempOrigins.length, reinterpret_cast<ShapeDescriptor::OrientedPoint*>(tempOrigins.content)};
    }




    std::cout << "Generating images.. (this can take a while)" << std::endl;
    if(generationMode.value() == "si") {
        ShapeDescriptor::gpu::PointCloud pointCloud = ShapeDescriptor::utilities::sampleMesh(deviceMesh, spinImageSampleCount.value(), 0);

        ShapeDescriptor::gpu::array<ShapeDescriptor::SpinImageDescriptor> descriptors = ShapeDescriptor::gpu::generateSpinImages(
                pointCloud,
                deviceSpinOrigins,
                spinImageWidth.value(),
                supportAngle.value());
        std::cout << "Dumping results.. " << std::endl;
        ShapeDescriptor::cpu::array<ShapeDescriptor::SpinImageDescriptor> hostDescriptors = ShapeDescriptor::copy::deviceArrayToHost<ShapeDescriptor::SpinImageDescriptor>(descriptors);
        ShapeDescriptor::dump::descriptors(hostDescriptors, outputFile.value(), enableLogarithmicImage.value(), imagesPerRow.value());

        ShapeDescriptor::free::array<ShapeDescriptor::SpinImageDescriptor>(descriptors);
        delete[] hostDescriptors.content;

    } else if(generationMode.value() == "rici") {
        ShapeDescriptor::cpu::array<ShapeDescriptor::RICIDescriptor> hostDescriptors;
        if(generationDevice.value() == "gpu") {
            ShapeDescriptor::gpu::array<ShapeDescriptor::RICIDescriptor> descriptors =
                    ShapeDescriptor::gpu::generateRadialIntersectionCountImages(
                            deviceMesh,
                            deviceSpinOrigins,
                            spinImageWidth.value());
            hostDescriptors = ShapeDescriptor::copy::deviceArrayToHost(descriptors);
            ShapeDescriptor::free::array(descriptors);
        } else if(generationDevice.value() == "cpu") {
            hostDescriptors = ShapeDescriptor::cpu::generateRadialIntersectionCountImages(mesh, spinOrigins, spinImageWidth.value());
        }


        std::cout << "Dumping results.. " << std::endl;

        if(imageLimit.value() != -1) {
            hostDescriptors.length = std::min<int>(hostDescriptors.length, imageLimit.value());
        }
        ShapeDescriptor::dump::descriptors(hostDescriptors, outputFile.value(), enableLogarithmicImage.value(), imagesPerRow.value());
        delete[] hostDescriptors.content;



    } else if(generationMode.value() == "quicci") {
        ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> hostDescriptors;

        if(generationDevice.value() == "gpu") {
            ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> images = ShapeDescriptor::gpu::generateQUICCImages(
                    deviceMesh,
                    deviceSpinOrigins,
                    spinImageWidth.value());
            hostDescriptors = ShapeDescriptor::copy::deviceArrayToHost(images);
            ShapeDescriptor::free::array(images);
        } else if(generationDevice.value() == "cpu") {
            hostDescriptors = ShapeDescriptor::cpu::generateQUICCImages(mesh, spinOrigins, spinImageWidth.value());
        }

        std::cout << "Dumping results.. " << std::endl;

        if(imageLimit.value() != -1) {
            hostDescriptors.length = std::min<int>(hostDescriptors.length, imageLimit.value());
        }

        ShapeDescriptor::dump::descriptors(hostDescriptors, outputFile.value(), imagesPerRow.value());

        ShapeDescriptor::free::array(hostDescriptors);
    } else {
        std::cerr << "Unrecognised image type: " << generationMode.value() << std::endl;
        std::cerr << "Should be either 'si', 'rici', or 'quicci'." << std::endl;
    }

    ShapeDescriptor::free::mesh(mesh);
    if(generationDevice.value() == "gpu") {
        ShapeDescriptor::gpu::freeMesh(deviceMesh);
    }
}
