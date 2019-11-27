#include <arrrgh.hpp>
#include <spinImage/cpu/index/types/Index.h>
#include <spinImage/cpu/index/IndexBuilder.h>

int main(int argc, const char** argv) {
    arrrgh::parser parser("indexbuilder", "Create indexes for QUICCI images.");
    const auto& indexFile = parser.add<std::string>(
            "index-directory", "The directory where the index should be stored.", '\0', arrrgh::Required, "");
    const auto& sourceDirectory = parser.add<std::string>(
            "quicci-dump-directory", "The directory where binary dump files of QUICCI images are stored that should be indexed.", '\0', arrrgh::Required, "");
    const auto& showHelp = parser.add<bool>(
            "help", "Show this help message.", 'h', arrrgh::Optional, false);

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

    std::cout << "Building index from files in " << sourceDirectory.value() << "..." << std::endl;
    SpinImage::index::build(sourceDirectory.value(), indexFile.value());
}