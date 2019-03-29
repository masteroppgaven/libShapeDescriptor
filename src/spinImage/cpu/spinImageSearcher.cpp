#include <cmath>
#include <algorithm>
#include <iostream>
#include "spinImageSearcher.h"
#include <spinImage/common/spinImageDistanceFunction.cuh>

bool compareSearchResults(const DescriptorSearchResult &a, const DescriptorSearchResult &b)
{
    return a.correlation > b.correlation;
}


template<typename pixelType>
float computeAveragePixelValue(pixelType* descriptors, size_t spinImageIndex)
{
    const unsigned int spinImageElementCount = spinImageWidthPixels * spinImageWidthPixels;

    float sum = 0;

    for (int y = 0; y < spinImageWidthPixels; y++)
    {
        for (int x = 0; x < spinImageWidthPixels; x ++)
        {
            float pixelValue = float(descriptors[spinImageIndex * spinImageElementCount + (y * spinImageWidthPixels + x)]);
            sum += pixelValue;
        }
    }

    return sum / float(spinImageElementCount);
}

std::vector<std::vector<DescriptorSearchResult>> computeCorrelations(
        array<spinImagePixelType> needleDescriptors,
        size_t needleImageCount,
        array<spinImagePixelType> haystackDescriptors,
        size_t haystackImageCount) {

    std::vector<std::vector<DescriptorSearchResult>> searchResults;
    searchResults.resize(needleImageCount);

    float* needleImageAverages = new float[needleImageCount];
    float* haystackImageAverages = new float[haystackImageCount];

    for(size_t i = 0; i < needleImageCount; i++) {
        needleImageAverages[i] = computeAveragePixelValue<spinImagePixelType>(needleDescriptors.content, i);
    }

    for(size_t i = 0; i < needleImageCount; i++) {
        haystackImageAverages[i] = computeAveragePixelValue<spinImagePixelType>(haystackDescriptors.content, i);
    }

#pragma omp parallel for
    for(size_t image = 0; image < needleImageCount; image++) {
        std::vector<DescriptorSearchResult> imageResults;
        float needleAverage = needleImageAverages[image];

        for(size_t haystackImage = 0; haystackImage < haystackImageCount; haystackImage++) {
            float haystackAverage = haystackImageAverages[haystackImage];

            float correlation = computeSpinImagePairCorrelationCPU(
                    needleDescriptors.content,
                    haystackDescriptors.content,
                    image, haystackImage,
                    needleAverage, haystackAverage);

            DescriptorSearchResult entry;
            entry.correlation = correlation;
            entry.imageIndex = haystackImage;
            imageResults.push_back(entry);
        }

        std::sort(imageResults.begin(), imageResults.end(), compareSearchResults);
        searchResults.at(image) = imageResults;
    }

    delete[] needleImageAverages;
    delete[] haystackImageAverages;

    std::cout << "Analysed " << searchResults.size() << " images on the CPU." << std::endl;

    return searchResults;
}

std::vector<std::vector<DescriptorSearchResult>> SpinImage::cpu::findDescriptorsInHaystack(
        array<spinImagePixelType> needleDescriptors,
        size_t needleImageCount,
        array<spinImagePixelType> haystackDescriptors,
        size_t haystackImageCount) {
    return computeCorrelations(needleDescriptors, needleImageCount, haystackDescriptors, haystackImageCount);
}

float SpinImage::cpu::computeImagePairCorrelation(spinImagePixelType* descriptors,
                                  spinImagePixelType* otherDescriptors,
                                  size_t spinImageIndex,
                                  size_t otherImageIndex) {
    float averageX = computeAveragePixelValue<spinImagePixelType>(descriptors, spinImageIndex);
    float averageY = computeAveragePixelValue<spinImagePixelType>(otherDescriptors, otherImageIndex);
    return computeSpinImagePairCorrelationCPU(descriptors, otherDescriptors, spinImageIndex, otherImageIndex, averageX, averageY);
}

float SpinImage::cpu::computeImageAverage(spinImagePixelType* descriptors, size_t spinImageIndex) {
    return computeAveragePixelValue<spinImagePixelType>(descriptors, spinImageIndex);
}

float SpinImage::cpu::computeImageAverage(quasiSpinImagePixelType* descriptors, size_t spinImageIndex) {
    return computeAveragePixelValue<quasiSpinImagePixelType>(descriptors, spinImageIndex);
}