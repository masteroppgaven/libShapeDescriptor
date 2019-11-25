#include <arrrgh.hpp>
#include <spinImage/cpu/types/Mesh.h>
#include <spinImage/utilities/OBJLoader.h>
#include <spinImage/utilities/copy/hostMeshToDevice.h>
#include <spinImage/gpu/types/Mesh.h>
#include <spinImage/gpu/types/QUICCImages.h>
#include <spinImage/gpu/types/DeviceOrientedPoint.h>
#include <spinImage/gpu/quickIntersectionCountImageGenerator.cuh>
#include <spinImage/utilities/duplicateRemoval.cuh>
#include <spinImage/gpu/radialIntersectionCountImageGenerator.cuh>
#include <spinImage/libraryBuildSettings.h>

const float DEFAULT_SPIN_IMAGE_WIDTH = 0.3;

int main(int argc, const char** argv) {
    arrrgh::parser parser("quiccidumper", "Render QUICCI images from an input OBJ file, and dump them to a file in binary form.");
    const auto& inputOBJFile = parser.add<std::string>("input-obj-file", "Location of the OBJ file from which the images should be rendered.", '\0', arrrgh::Required, "");
    const auto& outputDumpFile = parser.add<std::string>("output-dump-file", "Location where the generated images should be dumped to.", '\0', arrrgh::Required, "");
    const auto& fitInUnitSphere = parser.add<bool>("fit-object-in-unit-sphere", "Scale the object such that it fits in a unit sphere", '\0', arrrgh::Optional, false);
    const auto& spinImageWidth = parser.add<float>("spin-image-width", "The size of the spin image plane in 3D object space", '\0', arrrgh::Optional, DEFAULT_SPIN_IMAGE_WIDTH);
    const auto& showHelp = parser.add<bool>("help", "Show this help message.", 'h', arrrgh::Optional, false);

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

    SpinImage::cpu::Mesh hostMesh = SpinImage::utilities::loadOBJ(inputOBJFile.value(), true);
    SpinImage::gpu::Mesh deviceMesh = SpinImage::copy::hostMeshToDevice(hostMesh);
    SpinImage::cpu::freeMesh(hostMesh);

    SpinImage::array<SpinImage::gpu::DeviceOrientedPoint> uniqueVertices =
            SpinImage::utilities::computeUniqueVertices(deviceMesh);

    SpinImage::array<radialIntersectionCountImagePixelType> RICImages =
            SpinImage::gpu::generateRadialIntersectionCountImages(deviceMesh, uniqueVertices, spinImageWidth.value());

    SpinImage::gpu::QUICCIImages images = SpinImage::gpu::generateQUICCImages(RICImages);

    SpinImage::gpu::freeMesh(deviceMesh);

}