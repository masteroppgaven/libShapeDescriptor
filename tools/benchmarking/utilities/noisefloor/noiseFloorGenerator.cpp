#include "noiseFloorGenerator.h"
#include <shapeDescriptor/cpu/types/Mesh.h>
#include <shapeDescriptor/utilities/read/MeshLoader.h>
#include <shapeDescriptor/gpu/radialIntersectionCountImageSearcher.cuh>
#include <shapeDescriptor/gpu/quickIntersectionCountImageSearcher.cuh>
#include <shapeDescriptor/gpu/spinImageSearcher.cuh>
#include <shapeDescriptor/gpu/3dShapeContextSearcher.cuh>
#include <shapeDescriptor/gpu/fastPointFeatureHistogramSearcher.cuh>
#include <shapeDescriptor/utilities/free/mesh.h>
#include <shapeDescriptor/utilities/free/array.h>
#include <benchmarking/utilities/descriptor/RICI.h>
#include <benchmarking/utilities/descriptor/QUICCI.h>
#include <benchmarking/utilities/descriptor/spinImage.h>
#include <benchmarking/utilities/descriptor/3dShapeContext.h>
#include <benchmarking/utilities/descriptor/FPFH.h>
#include <benchmarking/utilities/metadata/generateFakeMetadata.h>
#include <benchmarking/utilities/metadata/transformDescriptor.h>
#include <json.hpp>
#include <vector>

using json = nlohmann::json;

template <typename T>
json distanceArrayToJson(T distanceArray)
{
    json distances;

    for (int i = 0; i < distanceArray.length; i++)
    {
        if (!isnan(distanceArray.content[i]))
            distances.push_back(distanceArray.content[i]);
    }

    return distances;
}

template <typename T>
json getDescriptorDistances(
    ShapeDescriptor::cpu::array<T> descriptorOne,
    ShapeDescriptor::cpu::array<T> descriptorTwo,
    size_t descriptorSampleCount = 200000,
    size_t correspondingDescriptorSampleCount = 200000)
{
    json output;

    ShapeDescriptor::gpu::array<T> descriptorOneGPU;
    ShapeDescriptor::gpu::array<T> descriptorTwoGPU;

    if (descriptorOne.length != descriptorTwo.length)
    {
        std::vector<std::variant<int, std::string>> metadata = Benchmarking::utilities::metadata::generateFakeMetadata(descriptorOne.length);
        std::vector<ShapeDescriptor::cpu::array<T>> transformedDescriptors = Benchmarking::utilities::metadata::transformDescriptorsToMatchMetadata(descriptorOne, descriptorTwo, metadata);
        descriptorOneGPU = transformedDescriptors[0].copyToGPU();
        descriptorTwoGPU = transformedDescriptors[1].copyToGPU();
    }
    else
    {
        descriptorOneGPU = descriptorOne.copyToGPU();
        descriptorTwoGPU = descriptorTwo.copyToGPU();
    }

    if constexpr (std::is_same_v<T, ShapeDescriptor::RICIDescriptor>)
    {
        ShapeDescriptor::cpu::array<int> crdDistances = ShapeDescriptor::gpu::computeRICIElementWiseModifiedSquareSumDistances(descriptorOneGPU, descriptorTwoGPU);
        output = distanceArrayToJson(crdDistances);
    }
    else if constexpr (std::is_same_v<T, ShapeDescriptor::QUICCIDescriptor>)
    {
        ShapeDescriptor::cpu::array<float> weightedHammingDistances = ShapeDescriptor::gpu::computeQUICCIElementWiseWeightedHammingDistances(descriptorOneGPU, descriptorTwoGPU);
        output = distanceArrayToJson(weightedHammingDistances);
    }
    else if constexpr (std::is_same_v<T, ShapeDescriptor::SpinImageDescriptor>)
    {
        ShapeDescriptor::cpu::array<float> pearsonCorrelations = ShapeDescriptor::gpu::computeSIElementWisePearsonCorrelations(descriptorOneGPU, descriptorTwoGPU);
        output = distanceArrayToJson(pearsonCorrelations);
    }
    else if constexpr (std::is_same_v<T, ShapeDescriptor::ShapeContextDescriptor>)
    {
        ShapeDescriptor::cpu::array<float> squaredDistances = ShapeDescriptor::gpu::compute3DSCElementWiseSquaredDistances(descriptorOneGPU, descriptorSampleCount, descriptorTwoGPU, correspondingDescriptorSampleCount);
        output = distanceArrayToJson(squaredDistances);
    }
    else if constexpr (std::is_same_v<T, ShapeDescriptor::FPFHDescriptor>)
    {
        ShapeDescriptor::cpu::array<float> euclideanDistances = ShapeDescriptor::gpu::computeFPFHElementWiseEuclideanDistances(descriptorOneGPU, descriptorTwoGPU);
        output = distanceArrayToJson(euclideanDistances);
    }

    ShapeDescriptor::free::array(descriptorOneGPU);
    ShapeDescriptor::free::array(descriptorTwoGPU);
    ShapeDescriptor::free::array(descriptorOne);
    ShapeDescriptor::free::array(descriptorTwo);

    return output;
}

json Benchmarking::utilities::noisefloor::generateNoiseFloor(
    std::string objectOnePath,
    std::string objectTwoPath,
    int descriptor,
    std::string hardware,
    float supportRadius,
    float supportAngleDegrees,
    float pointDensityRadius,
    float minSupportRadius,
    float maxSupportRadius,
    size_t pointCloudSampleCount,
    size_t randomSeed)
{
    json output;
    output["objectOnePath"] = objectOnePath;
    output["objectTwoPath"] = objectTwoPath;
    output["noiseFloors"] = {};

    ShapeDescriptor::cpu::Mesh meshOne = ShapeDescriptor::utilities::loadMesh(objectOnePath);
    ShapeDescriptor::cpu::Mesh meshTwo = ShapeDescriptor::utilities::loadMesh(objectTwoPath);

    switch (descriptor)
    {
    case 0:
    {
        std::chrono::duration<double> elapsedSecondsDescriptorOne;
        std::chrono::duration<double> elapsedSecondsDescriptorTwo;

        ShapeDescriptor::cpu::array<ShapeDescriptor::RICIDescriptor> descriptorsOne =
            Benchmarking::utilities::descriptor::generateRICIDescriptor(meshOne, hardware, supportRadius, elapsedSecondsDescriptorOne);
        ShapeDescriptor::cpu::array<ShapeDescriptor::RICIDescriptor> descriptorsTwo =
            Benchmarking::utilities::descriptor::generateRICIDescriptor(meshTwo, hardware, supportRadius, elapsedSecondsDescriptorTwo);

        output["noiseFloors"] = getDescriptorDistances(descriptorsOne, descriptorsTwo);
        break;
    }
    case 1:
    {
        std::chrono::duration<double> elapsedSecondsDescriptorOne;
        std::chrono::duration<double> elapsedSecondsDescriptorTwo;

        ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> descriptorsOne =
            Benchmarking::utilities::descriptor::generateQUICCIDescriptor(meshOne, hardware, supportRadius, elapsedSecondsDescriptorOne);
        ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> descriptorsTwo =
            Benchmarking::utilities::descriptor::generateQUICCIDescriptor(meshTwo, hardware, supportRadius, elapsedSecondsDescriptorTwo);

        output["noiseFloors"] = getDescriptorDistances(descriptorsOne, descriptorsTwo);
        break;
    }
    case 2:
    {
        std::chrono::duration<double> elapsedSecondsDescriptorOne;
        std::chrono::duration<double> elapsedSecondsDescriptorTwo;

        ShapeDescriptor::cpu::array<ShapeDescriptor::SpinImageDescriptor> descriptorsOne =
            Benchmarking::utilities::descriptor::generateSpinImageDescriptor(
                meshOne, hardware, supportRadius, supportAngleDegrees, pointCloudSampleCount, randomSeed, elapsedSecondsDescriptorOne);
        ShapeDescriptor::cpu::array<ShapeDescriptor::SpinImageDescriptor> descriptorsTwo =
            Benchmarking::utilities::descriptor::generateSpinImageDescriptor(
                meshTwo, hardware, supportRadius, supportAngleDegrees, pointCloudSampleCount, randomSeed, elapsedSecondsDescriptorTwo);

        output["noiseFloors"] = getDescriptorDistances(descriptorsOne, descriptorsTwo);
        break;
    }
    case 3:
    {
        std::chrono::duration<double> elapsedSecondsDescriptorOne;
        std::chrono::duration<double> elapsedSecondsDescriptorTwo;

        ShapeDescriptor::cpu::array<ShapeDescriptor::ShapeContextDescriptor> descriptorsOne =
            Benchmarking::utilities::descriptor::generate3DShapeContextDescriptor(
                meshOne, hardware, pointCloudSampleCount, randomSeed, pointDensityRadius, minSupportRadius, maxSupportRadius, elapsedSecondsDescriptorOne);
        ShapeDescriptor::cpu::array<ShapeDescriptor::ShapeContextDescriptor> descriptorsTwo =
            Benchmarking::utilities::descriptor::generate3DShapeContextDescriptor(
                meshTwo, hardware, pointCloudSampleCount, randomSeed, pointDensityRadius, minSupportRadius, maxSupportRadius, elapsedSecondsDescriptorTwo);

        output["noiseFloors"] = getDescriptorDistances(descriptorsOne, descriptorsTwo, pointCloudSampleCount, pointCloudSampleCount);
        break;
    }
    case 4:
    {
        std::chrono::duration<double> elapsedSecondsDescriptorOne;
        std::chrono::duration<double> elapsedSecondsDescriptorTwo;

        ShapeDescriptor::cpu::array<ShapeDescriptor::FPFHDescriptor> descriptorsOne =
            Benchmarking::utilities::descriptor::generateFPFHDescriptor(
                meshOne, hardware, supportRadius, pointCloudSampleCount, randomSeed, elapsedSecondsDescriptorOne);
        ShapeDescriptor::cpu::array<ShapeDescriptor::FPFHDescriptor> descriptorsTwo =
            Benchmarking::utilities::descriptor::generateFPFHDescriptor(
                meshTwo, hardware, supportRadius, pointCloudSampleCount, randomSeed, elapsedSecondsDescriptorTwo);

        output["noiseFloors"] = getDescriptorDistances(descriptorsOne, descriptorsTwo);
        break;
    }
    default:
    {
        std::cout << "Invalid descriptor type" << std::endl;
        break;
    }
    }

    ShapeDescriptor::free::mesh(meshOne);
    ShapeDescriptor::free::mesh(meshTwo);

    return output;
}