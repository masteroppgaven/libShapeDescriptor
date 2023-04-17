#include "transformDescriptor.h"
#include <vector>

int Benchmarking::utilities::metadata::getNumberOfDeletedVerticiesInMetadata(std::vector<std::variant<int, std::string>> metadata)
{
    int numberOfDeletedVerticies = 0;
    for (int i = 0; i < metadata.size(); i++)
    {
        try
        {
            std::get<int>(metadata[i]);
        }
        catch (std::exception &e)
        {
            numberOfDeletedVerticies++;
        }
    }

    return numberOfDeletedVerticies;
}