#include <shapeDescriptor/cpu/types/Mesh.h>
#include <shapeDescriptor/utilities/read/MeshLoader.h>
#include <shapeDescriptor/utilities/spinOriginsGenerator.h>
#include <shapeDescriptor/cpu/radialIntersectionCountImageGenerator.h>
#include <shapeDescriptor/cpu/quickIntersectionCountImageGenerator.h>
#include <shapeDescriptor/cpu/spinImageGenerator.h>
#include <shapeDescriptor/utilities/free/mesh.h>
#include <shapeDescriptor/utilities/free/array.h>
#include <shapeDescriptor/gpu/spinImageSearcher.cuh>
#include <shapeDescriptor/gpu/radialIntersectionCountImageSearcher.cuh>
#include <shapeDescriptor/common/types/methods/SpinImageDescriptor.h>
#include <benchmarking/utilities/descriptor/RICI.h>
#include <benchmarking/utilities/descriptor/QUICCI.h>
#include <benchmarking/utilities/descriptor/spinImage.h>
#include <benchmarking/utilities/descriptor/3dShapeContext.h>
#include <benchmarking/utilities/descriptor/FPFH.h>
#include <benchmarking/utilities/distance/similarity.h>
#include <benchmarking/utilities/metadata/generateFakeMetadata.h>
#include <benchmarking/utilities/metadata/prepareMetadata.h>
#include <benchmarking/utilities/metadata/transformDescriptor.h>
#include <benchmarking/utilities/noisefloor/noiseFloorGenerator.h>
#include <tuple>
#include <iostream>
#include <fstream>
#include <arrrgh.hpp>
#include <vector>
#include <variant>
#include <map>
#include <json.hpp>
#include <ctime>
#include <chrono>
#include <git.h>
#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
#include <cuda_runtime_api.h>
#endif

using json = nlohmann::json;

json originalObjectsData;

using descriptorType = std::variant<
    ShapeDescriptor::cpu::array<ShapeDescriptor::RICIDescriptor>,
    ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor>,
    ShapeDescriptor::cpu::array<ShapeDescriptor::SpinImageDescriptor>,
    ShapeDescriptor::cpu::array<ShapeDescriptor::ShapeContextDescriptor>,
    ShapeDescriptor::cpu::array<ShapeDescriptor::FPFHDescriptor>>;

std::map<int, std::string> descriptorAlgorithms = {
    {0, "RICI"},
    {1, "QUICCI"},
    {2, "SI"},
    {3, "3DSC"},
    {4, "FPFH"}};

struct
{
    std::string name;
    int clockRate;
    int memory;
} GPUInfo;

const auto runDate = std::chrono::system_clock::now();

std::string getRunDate()
{
    auto in_time_t = std::chrono::system_clock::to_time_t(runDate);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d-%H:%M:%S");
    return ss.str();
}

template <typename T>
float calculateAverageSimilartyFromDistancesArray(ShapeDescriptor::cpu::array<T> distances)
{
    float simSum = 0;

    for (int i = 0; i < distances.length; i++)
    {
        float sim = 1 / (1 + distances[i]);
        if (!isnan(sim))
            simSum += sim;
    }
    float avgSim = simSum / distances.length;

    return avgSim;
}

template <typename T>
float calculateAverageSimilarity(ShapeDescriptor::cpu::array<T> distances)
{
    float simSum = 0;

    for (int i = 0; i < distances.length; i++)
    {
        simSum += isnan(distances[i]) ? 0 : distances[i];
    }

    float avgSim = simSum / distances.length;

    return avgSim;
}

template <typename T>
double calculateSimilarity(ShapeDescriptor::cpu::array<T> dOriginal, ShapeDescriptor::cpu::array<T> dComparison, int distanceFunction, bool freeArray)
{
    double sim = Benchmarking::utilities::distance::similarityBetweenTwoDescriptors<T>(dOriginal, dComparison, distanceFunction);

    if (freeArray)
    {
        ShapeDescriptor::free::array(dOriginal);
        ShapeDescriptor::free::array(dComparison);
    }

    return sim;
}

descriptorType generateDescriptorsForObject(ShapeDescriptor::cpu::Mesh mesh,
                                            int algorithm,
                                            std::string hardware,
                                            std::chrono::duration<double> &elapsedTime,
                                            float supportRadius = 2.5f,
                                            float supportAngleDegrees = 60.0f,
                                            float pointDensityRadius = 0.2f,
                                            float minSupportRadius = 0.1f,
                                            float maxSupportRadius = 2.5f,
                                            size_t pointCloudSampleCount = 200000,
                                            size_t randomSeed = 4917133789385064)
{
    descriptorType descriptor;

    switch (algorithm)
    {
    case 0:
    {
        descriptor = Benchmarking::utilities::descriptor::generateRICIDescriptor(
            mesh, hardware, supportRadius, elapsedTime);
        break;
    }
    case 1:
    {
        descriptor = Benchmarking::utilities::descriptor::generateQUICCIDescriptor(
            mesh, hardware, supportRadius, elapsedTime);
        break;
    }
    case 2:
    {
        descriptor = Benchmarking::utilities::descriptor::generateSpinImageDescriptor(
            mesh, hardware, supportRadius, supportAngleDegrees, pointCloudSampleCount, randomSeed, elapsedTime);
        break;
    }
    case 3:
    {
        descriptor = Benchmarking::utilities::descriptor::generate3DShapeContextDescriptor(
            mesh, hardware, pointCloudSampleCount, randomSeed, pointDensityRadius, minSupportRadius, maxSupportRadius, elapsedTime);
        break;
    }
    case 4:
    {
        descriptor = Benchmarking::utilities::descriptor::generateFPFHDescriptor(
            mesh, hardware, supportRadius, pointCloudSampleCount, randomSeed, elapsedTime);
        break;
    }
    default:
    {
        descriptor = Benchmarking::utilities::descriptor::generateRICIDescriptor(
            mesh, hardware, supportRadius, elapsedTime);
        break;
    }
    };

    return descriptor;
}

template <typename T>
void freeDescriptorType(ShapeDescriptor::cpu::array<T> descriptor)
{
    ShapeDescriptor::free::array(descriptor);
}

int getNumberOfFilesInFolder(std::string folderPath)
{
    int numberOfFiles = 0;
    for (const auto &entry : std::filesystem::directory_iterator(folderPath))
    {
        numberOfFiles++;
    }

    return numberOfFiles;
}

void multipleObjectsBenchmark(
    std::string objectsFolder,
    std::string originalsFolderName,
    std::string jsonPath,
    std::string hardware,
    std::string compareFolder,
    std::string previousRunPath,
    float supportRadius = 2.5f,
    float supportAngleDegrees = 60.0f,
    float pointDensityRadius = 0.2f,
    float minSupportRadius = 0.1f,
    float maxSupportRadius = 2.5f,
    size_t pointCloudSampleCount = 200000,
    size_t randomSeed = 4917133789385064)
{
    std::vector<std::string> folders;
    std::string originalObjectFolderPath;

    json previousRun;
    if (std::filesystem::exists(previousRunPath))
    {
        std::ifstream jsonFile(previousRunPath);
        previousRun = json::parse(jsonFile);
    }

    // This is hard coded for now, as this fits how we have structured the folder. Should be edited if you want the code more dynamic:^)
    std::string originalObjectCategory = "0-100";

    std::string outputDirectory = jsonPath + "/" + getRunDate();

    for (auto &p : std::filesystem::directory_iterator(objectsFolder))
    {
        if (p.is_directory())
        {
            if (originalObjectFolderPath.empty() && p.path().string().substr(p.path().string().find_last_of("/") + 1) == originalsFolderName)
            {
                originalObjectFolderPath = p.path().string() + "/" + originalObjectCategory;
            }
            else if (compareFolder == "")
            {
                folders.push_back(p.path().string());
            }
            else if (p.path().string().substr(p.path().string().find_last_of("/") + 1) == compareFolder)
            {
                folders.push_back(p.path().string());
            }
        }
    }

    for (std::string folder : folders)
    {
        std::chrono::steady_clock::time_point timeStart = std::chrono::steady_clock::now();
        json jsonOutput;
        std::string comparisonFolderName = folder.substr(folder.find_last_of("/") + 1);

        if (!std::filesystem::exists(outputDirectory + "-" + comparisonFolderName))
        {
            std::filesystem::create_directory(outputDirectory + "-" + comparisonFolderName);
        }

        jsonOutput["runDate"] = getRunDate();
        jsonOutput["hardware"]["type"] = hardware;

        jsonOutput["buildInfo"]["commit"] = GitMetadata::CommitSHA1();
        jsonOutput["buildInfo"]["commit_author"] = GitMetadata::AuthorEmail();
        jsonOutput["buildInfo"]["commit_date"] = GitMetadata::CommitSubject();

        jsonOutput["static"] = {};
        jsonOutput["static"]["supportRadius"] = supportRadius;
        jsonOutput["static"]["supportAngleDegrees"] = supportAngleDegrees;
        jsonOutput["static"]["pointDensityRadius"] = pointDensityRadius;
        jsonOutput["static"]["minSupportRadius"] = minSupportRadius;
        jsonOutput["static"]["maxSupportRadius"] = maxSupportRadius;
        jsonOutput["static"]["pointCloudSampleCount"] = pointCloudSampleCount;
        jsonOutput["static"]["randomSeed"] = randomSeed;

#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
        jsonOutput["hardware"]["gpu"]["name"] = GPUInfo.name;
        jsonOutput["hardware"]["gpu"]["clockRate"] = GPUInfo.clockRate;
        jsonOutput["hardware"]["gpu"]["memory"] = GPUInfo.memory;
#endif

        for (auto &categoryPath : std::filesystem::directory_iterator(folder))
        {
            std::string category = categoryPath.path().string().substr(categoryPath.path().string().find_last_of("/") + 1);

            if (!categoryPath.is_directory())
                continue;

            for (auto &originalObject : std::filesystem::directory_iterator(originalObjectFolderPath))
            {
                std::string originalFolder = originalObject.path().string();
                std::string fileName = originalFolder.substr(originalFolder.find_last_of("/") + 1);

                std::filesystem::path originalObjectPath;
                std::filesystem::path comparisonObjectPath;
                std::vector<std::variant<int, std::string>> metadata;

                ShapeDescriptor::cpu::Mesh meshOriginal;
                ShapeDescriptor::cpu::Mesh meshComparison;

                std::string comparisonFolder = folder + "/" + category + "/" + fileName;

                std::cout << "Comparing object " << fileName << " in category " << category << std::endl;

                try
                {
                    originalObjectPath = originalFolder + "/" + fileName + ".obj";
                    comparisonObjectPath = comparisonFolder + "/" + fileName + ".obj";

                    meshOriginal = ShapeDescriptor::utilities::loadMesh(originalObjectPath);
                    meshComparison = ShapeDescriptor::utilities::loadMesh(comparisonObjectPath);

                    metadata = Benchmarking::utilities::metadata::prepareMetadata(comparisonFolder + "/" + fileName + ".txt", meshOriginal.vertexCount);
                }
                catch (const std::exception e)
                {
                    std::cout << "Comparison object " << comparisonObjectPath << " not found..." << std::endl;
                    continue;
                }

                jsonOutput["results"][fileName][comparisonFolderName]["vertexCounts"][category]["vertexCount"] = meshComparison.vertexCount;

                for (auto a : descriptorAlgorithms)
                {
                    std::chrono::duration<double> elapsedSecondsDescriptorComparison;
                    std::chrono::duration<double> elapsedSecondsDescriptorOriginal;

                    if (previousRun["results"][fileName][comparisonFolderName][a.second][category].size() > 0)
                    {
                        jsonOutput["results"][fileName][comparisonFolderName][a.second][category] = previousRun["results"][fileName][comparisonFolderName][a.second][category];
                        continue;
                    }

                    descriptorType originalObject = generateDescriptorsForObject(
                        meshOriginal, a.first, hardware, elapsedSecondsDescriptorOriginal,
                        supportRadius, supportAngleDegrees, pointDensityRadius, minSupportRadius, maxSupportRadius,
                        pointCloudSampleCount, randomSeed);

                    descriptorType comparisonObject = generateDescriptorsForObject(
                        meshComparison, a.first, hardware, elapsedSecondsDescriptorComparison,
                        supportRadius, supportAngleDegrees, pointDensityRadius, minSupportRadius, maxSupportRadius,
                        pointCloudSampleCount, randomSeed);

                    std::chrono::steady_clock::time_point distanceTimeStart;
                    std::chrono::steady_clock::time_point distanceTimeEnd;

                    std::string distanceFunction = "";

                    switch (a.first)
                    {
                    case 0:
                    {
                        std::vector<ShapeDescriptor::cpu::array<ShapeDescriptor::RICIDescriptor>> transformed =
                            Benchmarking::utilities::metadata::transformDescriptorsToMatchMetadata(std::get<0>(originalObject), std::get<0>(comparisonObject), metadata);

                        freeDescriptorType(std::get<0>(originalObject));
                        freeDescriptorType(std::get<0>(comparisonObject));

                        ShapeDescriptor::cpu::array<ShapeDescriptor::RICIDescriptor> original =
                            transformed.at(0);

                        ShapeDescriptor::cpu::array<ShapeDescriptor::RICIDescriptor> comparison =
                            transformed.at(1);

                        distanceFunction = "Clutter Resistant Distance";

                        if (originalObjectsData["results"].find(fileName) == originalObjectsData["results"].end())
                        {
                            originalObjectsData["results"][fileName][a.second]["generationTime"] = elapsedSecondsDescriptorOriginal.count();
                            // The length of the descriptor is the exact number of verticies
                            // While the vertex count in the mesh class is just faces * 3, which is can sometimes be not accurate
                            originalObjectsData["results"][fileName]["vertexCount"] = original.length;
                        }

                        distanceTimeStart = std::chrono::steady_clock::now();
                        // Hente alle distanser, og regne ut gjennomsnitt
                        // sim = calculateSimilarity<ShapeDescriptor::RICIDescriptor>(original, comparison, d.first, freeArray);
                        distanceTimeEnd = std::chrono::steady_clock::now();
                        break;
                    }
                    case 1:
                    {
                        std::vector<ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor>> transformed =
                            Benchmarking::utilities::metadata::transformDescriptorsToMatchMetadata(std::get<1>(originalObject), std::get<1>(comparisonObject), metadata);

                        freeDescriptorType(std::get<1>(originalObject));
                        freeDescriptorType(std::get<1>(comparisonObject));

                        ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> original =
                            transformed.at(0);

                        ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> comparison =
                            transformed.at(1);

                        distanceFunction = "Weighted Hamming Distance";

                        if (originalObjectsData["results"].find(fileName) == originalObjectsData["results"].end())
                        {
                            originalObjectsData["results"][fileName][a.second]["generationTime"] = elapsedSecondsDescriptorOriginal.count();
                            originalObjectsData["results"][fileName]["vertexCount"] = original.length;
                        }

                        distanceTimeStart = std::chrono::steady_clock::now();
                        // sim = calculateSimilarity<ShapeDescriptor::QUICCIDescriptor>(original, comparison, d.first, freeArray);
                        distanceTimeEnd = std::chrono::steady_clock::now();
                        break;
                    }
                    case 2:
                    {
                        std::vector<ShapeDescriptor::cpu::array<ShapeDescriptor::SpinImageDescriptor>> transformed =
                            Benchmarking::utilities::metadata::transformDescriptorsToMatchMetadata(std::get<2>(originalObject), std::get<2>(comparisonObject), metadata);

                        freeDescriptorType(std::get<2>(originalObject));
                        freeDescriptorType(std::get<2>(comparisonObject));

                        ShapeDescriptor::cpu::array<ShapeDescriptor::SpinImageDescriptor> original =
                            transformed.at(0);

                        ShapeDescriptor::cpu::array<ShapeDescriptor::SpinImageDescriptor> comparison =
                            transformed.at(1);

                        distanceFunction = "Pearson Correlation Coeffecient";

                        if (originalObjectsData["results"].find(fileName) == originalObjectsData["results"].end())
                        {
                            originalObjectsData["results"][fileName][a.second]["generationTime"] = elapsedSecondsDescriptorOriginal.count();
                            originalObjectsData["results"][fileName]["vertexCount"] = original.length;
                        }

                        distanceTimeStart = std::chrono::steady_clock::now();
                        // sim = calculateSimilarity<ShapeDescriptor::SpinImageDescriptor>(original, comparison, d.first, freeArray);
                        distanceTimeEnd = std::chrono::steady_clock::now();
                        break;
                    }
                    case 3:
                    {
                        std::vector<ShapeDescriptor::cpu::array<ShapeDescriptor::ShapeContextDescriptor>> transformed =
                            Benchmarking::utilities::metadata::transformDescriptorsToMatchMetadata(std::get<3>(originalObject), std::get<3>(comparisonObject), metadata);

                        freeDescriptorType(std::get<3>(originalObject));
                        freeDescriptorType(std::get<3>(comparisonObject));

                        ShapeDescriptor::cpu::array<ShapeDescriptor::ShapeContextDescriptor> original =
                            transformed.at(0);

                        ShapeDescriptor::cpu::array<ShapeDescriptor::ShapeContextDescriptor> comparison =
                            transformed.at(1);

                        distanceFunction = "Euclidean Distance";

                        if (originalObjectsData["results"].find(fileName) == originalObjectsData["results"].end())
                        {
                            originalObjectsData["results"][fileName][a.second]["generationTime"] = elapsedSecondsDescriptorOriginal.count();
                            originalObjectsData["results"][fileName]["vertexCount"] = original.length;
                        }

                        distanceTimeStart = std::chrono::steady_clock::now();
                        // sim = calculateSimilarity<ShapeDescriptor::ShapeContextDescriptor>(original, comparison, d.first, freeArray);
                        distanceTimeEnd = std::chrono::steady_clock::now();
                        break;
                    }
                    case 4:
                    {
                        std::vector<ShapeDescriptor::cpu::array<ShapeDescriptor::FPFHDescriptor>> transformed =
                            Benchmarking::utilities::metadata::transformDescriptorsToMatchMetadata(std::get<4>(originalObject), std::get<4>(comparisonObject), metadata);

                        freeDescriptorType(std::get<4>(originalObject));
                        freeDescriptorType(std::get<4>(comparisonObject));

                        ShapeDescriptor::cpu::array<ShapeDescriptor::FPFHDescriptor> original =
                            transformed.at(0);

                        ShapeDescriptor::cpu::array<ShapeDescriptor::FPFHDescriptor> comparison =
                            transformed.at(1);

                        distanceFunction = "Euclidean Distance";

                        if (originalObjectsData["results"].find(fileName) == originalObjectsData["results"].end())
                        {
                            originalObjectsData["results"][fileName][a.second]["generationTime"] = elapsedSecondsDescriptorOriginal.count();
                            originalObjectsData["results"][fileName]["vertexCount"] = original.length;
                        }
                        distanceTimeStart = std::chrono::steady_clock::now();
                        // sim = calculateSimilarity<ShapeDescriptor::FPFHDescriptor>(original, comparison, d.first, freeArray);
                        distanceTimeEnd = std::chrono::steady_clock::now();
                        break;
                    }
                    default:
                    {
                        std::vector<ShapeDescriptor::cpu::array<ShapeDescriptor::RICIDescriptor>> transformed =
                            Benchmarking::utilities::metadata::transformDescriptorsToMatchMetadata(std::get<0>(originalObject), std::get<0>(comparisonObject), metadata);

                        freeDescriptorType(std::get<0>(originalObject));
                        freeDescriptorType(std::get<0>(comparisonObject));

                        ShapeDescriptor::cpu::array<ShapeDescriptor::RICIDescriptor> original =
                            transformed.at(0);

                        ShapeDescriptor::cpu::array<ShapeDescriptor::RICIDescriptor> comparison =
                            transformed.at(1);

                        distanceFunction = "Clutter Resistant Distance";

                        if (originalObjectsData["results"].find(fileName) == originalObjectsData["results"].end())
                        {
                            originalObjectsData["results"][fileName][a.second]["generationTime"] = elapsedSecondsDescriptorOriginal.count();
                            // The length of the descriptor is the exact number of verticies
                            // While the vertex count in the mesh class is just faces * 3, which is can sometimes be not accurate
                            originalObjectsData["results"][fileName]["vertexCount"] = original.length;
                        }

                        distanceTimeStart = std::chrono::steady_clock::now();
                        // Hente alle distanser, og regne ut gjennomsnitt
                        // sim = calculateSimilarity<ShapeDescriptor::RICIDescriptor>(original, comparison, d.first, freeArray);
                        distanceTimeEnd = std::chrono::steady_clock::now();
                        break;
                    }
                    }

                    std::chrono::duration<double> elapsedSecondsDistance = distanceTimeEnd - distanceTimeStart;

                    jsonOutput["results"][fileName][comparisonFolderName][a.second][category]["generationTime"] = elapsedSecondsDescriptorComparison.count();

                    // Kan bli average distanse
                    jsonOutput["results"][fileName][comparisonFolderName][a.second][category][distanceFunction]["similarity"] = (double)0.0;
                    jsonOutput["results"][fileName][comparisonFolderName][a.second][category][distanceFunction]["time"] = elapsedSecondsDistance.count();
                }
                ShapeDescriptor::free::mesh(meshOriginal);
                ShapeDescriptor::free::mesh(meshComparison);

                metadata.clear();

                std::chrono::steady_clock::time_point timeAfter = std::chrono::steady_clock::now();
                std::chrono::duration<double> currentTotalRunTime = timeAfter - timeStart;

                jsonOutput["runTime"] = currentTotalRunTime.count();

                std::string outputFilePath = outputDirectory + "-" + comparisonFolderName + "/" + comparisonFolderName + ".json";
                std::ofstream outFile(outputFilePath);
                outFile << jsonOutput.dump(4);
                outFile.close();

                std::cout << "Results stored to " << outputFilePath << std::endl;
            }
        }
    }
}

int main(int argc, const char **argv)
{
    arrrgh::parser parser("benchmarking", "Compare how similar two objects are (only OBJ file support)");
    const auto &originalObject = parser.add<std::string>("original-object", "Original object.", 'o', arrrgh::Optional, "");
    const auto &comparisonObject = parser.add<std::string>("comparison-object", "Object to compare to the original.", 'c', arrrgh::Optional, "");
    const auto &objectsFolder = parser.add<std::string>("objects-folder", "Folder consisting of sub-directories with all the different objects and their metadata", 'f', arrrgh::Optional, "");
    const auto &originalsFolderName = parser.add<std::string>("originals-folder", "Folder name with all the original objects (for example, RecalculatedNormals)", 'n', arrrgh::Optional, "RecalculatedNormals");
    const auto &compareFolder = parser.add<std::string>("compare-folder", "If you only want to compare the originals to a specific folder (for example, ObjectsWithHoles)", 'F', arrrgh::Optional, "");
    const auto &previousRunFile = parser.add<std::string>("previous-run", "Path to a JSON file containing data from a previous run", 'P', arrrgh::Optional, "");
    const auto &metadataPath = parser.add<std::string>("metadata", "Path to metadata describing which vertecies that are changed", 'm', arrrgh::Optional, "");
    const auto &outputPath = parser.add<std::string>("output-path", "Path to the output", 'p', arrrgh::Optional, "");
    const auto &descriptorAlgorithm = parser.add<int>("descriptor-algorithm", "Which descriptor algorithm to use [0 for radial-intersection-count-images, 1 for quick-intersection-count-change-images ...will add more:)]", 'a', arrrgh::Optional, 0);
    const auto &distanceAlgorithm = parser.add<int>("distance-algorithm", "Which distance algorithm to use [0 for euclidian, ...will add more:)]", 'd', arrrgh::Optional, 0);
    const auto &hardware = parser.add<std::string>("hardware-type", "cpu or gpu (gpu is default, as cpu doesn't support all the descriptors)", 't', arrrgh::Optional, "gpu");
    const auto &noisefloor = parser.add<int>("noisefloor", "Integer representing if you want to generate noisefloor or not, (0 for false, 1 for true)", 'N', arrrgh::Optional, 0);
    const auto &noisefloorObjects = parser.add<std::string>("noisefloor-objects", "Path to the folder containing the noisefloor objects", 'O', arrrgh::Optional, "");
    const auto &help = parser.add<bool>("help", "Show help", 'h', arrrgh::Optional, false);

    try
    {
        parser.parse(argc, argv);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error parsing arguments: " << e.what() << std::endl;
        parser.show_usage(std::cerr);
        exit(1);
    }

    if (help.value())
    {
        return 0;
    }

    float supportRadius = 1.5f;
    float supportAngleDegrees = 60.0f;
    float pointDensityRadius = 0.2f;
    float minSupportRadius = 0.1f;
    float maxSupportRadius = 2.5f;
    size_t pointCloudSampleCount = 200000;
    size_t randomSeed = 4917133789385064;

#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
    cudaDeviceProp device_information;
    cudaGetDeviceProperties(&device_information, 0);
    GPUInfo.name = std::string(device_information.name);
    GPUInfo.clockRate = device_information.clockRate;
    GPUInfo.memory = device_information.totalGlobalMem / (1024 * 1024);
#endif

    if (originalObject.value() != "" && comparisonObject.value() != "")
    {
        std::cout << "runDate " << getRunDate() << std::endl;

        int timeStart = std::time(0);
        std::filesystem::path objectOne = originalObject.value();
        std::filesystem::path objectTwo = comparisonObject.value();

        ShapeDescriptor::cpu::Mesh meshOne = ShapeDescriptor::utilities::loadMesh(objectOne);
        ShapeDescriptor::cpu::Mesh meshTwo = ShapeDescriptor::utilities::loadMesh(objectTwo);

        std::vector<std::variant<int, std::string>> metadata;

        if (metadataPath.value() == "")
        {
            metadata = Benchmarking::utilities::metadata::prepareMetadata("", meshOne.vertexCount);
        }
        else
        {
            metadata = Benchmarking::utilities::metadata::prepareMetadata(metadataPath.value());
        }

        std::chrono::duration<double> elapsedTimeOne;
        std::chrono::duration<double> elapsedTimeTwo;

        descriptorType descriptorOne = generateDescriptorsForObject(meshOne, 3, hardware.value(), elapsedTimeOne);
        descriptorType descriptorTwo = generateDescriptorsForObject(meshTwo, 3, hardware.value(), elapsedTimeTwo);

        ShapeDescriptor::free::mesh(meshOne);
        ShapeDescriptor::free::mesh(meshTwo);
    }
    else if (noisefloor.value() == 1)
    {
        json noiseFloors;
        std::vector<std::string> noisefloorObjectPaths;

        int timeStart = std::time(0);

        int numberOfObjects = 100;
        std::string objectsFolder = "/mnt/VOID/projects/shape_descriptors_benchmark/Dataset/NewRecalculatedNormals/0-100/";

        for (int object = 0; object < numberOfObjects; object++)
        {
            std::string objectStr = std::to_string(object);
            std::string objectName = std::string(4 - objectStr.length(), '0') + objectStr;

            std::filesystem::path objectPath = objectsFolder + objectName + "/" + objectName + ".obj";

            std::cout << "Generating noise roof for " << objectPath << std::endl;

            noiseFloors[objectName + ".obj"][objectName + ".obj"] =
                Benchmarking::utilities::noisefloor::generateNoiseFloor(
                    objectPath,
                    objectPath,
                    descriptorAlgorithm.value(),
                    hardware.value(),
                    supportRadius,
                    supportAngleDegrees,
                    pointDensityRadius,
                    minSupportRadius,
                    maxSupportRadius,
                    pointCloudSampleCount,
                    randomSeed);
        }

        noiseFloors["runDate"] = getRunDate();
        noiseFloors["static"] = {};
        noiseFloors["static"]["supportRadius"] = supportRadius;
        noiseFloors["static"]["supportAngleDegrees"] = supportAngleDegrees;
        noiseFloors["static"]["pointDensityRadius"] = pointDensityRadius;
        noiseFloors["static"]["minSupportRadius"] = minSupportRadius;
        noiseFloors["static"]["maxSupportRadius"] = maxSupportRadius;
        noiseFloors["static"]["pointCloudSampleCount"] = pointCloudSampleCount;
        noiseFloors["static"]["randomSeed"] = randomSeed;
        noiseFloors["shapeDescriptor"] = descriptorAlgorithm.value();

        std::string noiseOutputPath = outputPath.value() + "/" + std::to_string(descriptorAlgorithm.value()) + "-noiseroof-" + getRunDate() + ".json";
        std::ofstream noiseOut(noiseOutputPath);
        noiseOut << noiseFloors.dump(4);
        noiseOut.close();
    }
    else if (objectsFolder.value() != "" && originalsFolderName.value() != "" && (originalObject.value() == "" && comparisonObject.value() == ""))
    {
        std::cout << "Comparing all objects in folder..." << std::endl;
        multipleObjectsBenchmark(
            objectsFolder.value(),
            originalsFolderName.value(),
            outputPath.value(),
            hardware.value(),
            compareFolder.value(),
            previousRunFile.value(),
            supportRadius,
            supportAngleDegrees,
            pointDensityRadius,
            minSupportRadius,
            maxSupportRadius,
            pointCloudSampleCount,
            randomSeed);

        std::string originalObjectsDataPath = outputPath.value() + "/" + getRunDate() + "-" + originalsFolderName.value() + "/" + originalsFolderName.value() + ".json";

        std::cout << "Writing original objects data to " << originalObjectsDataPath << std::endl;

        std::ofstream outFile(originalObjectsDataPath);
        outFile << originalObjectsData.dump(4);
        outFile.close();
    }
    else
    {
        std::cout << "Wrong inputs, exiting..." << std::endl;
    }

    return 0;
}