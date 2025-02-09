#pragma once

#include <filesystem>
#include <vector>

namespace ShapeDescriptor {
    namespace utilities {
        std::vector<std::filesystem::path> listDirectory(const std::string& directory);
        void writeCompressedFile(const char* buffer, size_t bufferSize, const std::filesystem::path &archiveFile, unsigned int threadCount = 1);
        const char* readCompressedFile(const std::filesystem::path &archiveFile, size_t* fileSizeBytes, unsigned int threadCount = 1);
        const char* readCompressedFileUpToNBytes(const std::filesystem::path &archiveFile, size_t* readByteCount, size_t decompressedBytesToRead, unsigned int threadCount = 1);
        std::string generateUniqueFilenameString();
    }
}



