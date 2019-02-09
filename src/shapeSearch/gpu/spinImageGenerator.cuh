#pragma once

#include <shapeSearch/libraryBuildSettings.h>
#include <shapeSearch/gpu/types/DeviceMesh.h>

array<classicSpinImagePixelType> createClassicDescriptors(DeviceMesh device_mesh, cudaDeviceProp device_information, size_t sampleCount);