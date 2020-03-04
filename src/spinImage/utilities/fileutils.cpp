#include <fstream>
#include <cassert>
#include "fileutils.h"
#include <fast-lzma2.h>

const int LZMA2_COMPRESSION_LEVEL = 9;

std::vector<std::experimental::filesystem::path> SpinImage::utilities::listDirectory(const std::string& directory) {
    std::vector<std::experimental::filesystem::path> foundFiles;

    for(auto &path : std::experimental::filesystem::directory_iterator(directory)) {
        foundFiles.emplace_back(path);
    }

    return foundFiles;
}

const char *SpinImage::utilities::readCompressedFile(const std::experimental::filesystem::path &archiveFile, size_t* fileSizeBytes) {
    std::array<char, 5> headerTitle = {0, 0, 0, 0, 0};
    size_t compressedBufferSize;
    size_t decompressedBufferSize;

    std::ifstream decompressStream(archiveFile.string(), std::ios::out | std::ios::binary);

    decompressStream.read(headerTitle.data(), 5);
    decompressStream.read((char*) &decompressedBufferSize, sizeof(size_t));
    decompressStream.read((char*) &compressedBufferSize, sizeof(size_t));

    *fileSizeBytes = decompressedBufferSize;

    char* compressedBuffer = new char[compressedBufferSize];
    char* decompressedBuffer = new char[decompressedBufferSize];

    assert(std::string(headerTitle.data()) == "CDXF");

    decompressStream.read(compressedBuffer, compressedBufferSize);

    FL2_decompressMt(
            (void*) decompressedBuffer, decompressedBufferSize,
            (void*) compressedBuffer, compressedBufferSize, 6);

    delete[] compressedBuffer;

    return decompressedBuffer;
}

void SpinImage::utilities::writeCompressedFile(const char *buffer, size_t bufferSize, const std::experimental::filesystem::path &archiveFile) {

    std::experimental::filesystem::create_directories(archiveFile.parent_path());

    const size_t maxCompressedBufferSize = FL2_compressBound(bufferSize);
    char* compressedBuffer = new char[maxCompressedBufferSize];
    unsigned long compressedBufferSize =
            FL2_compressMt(
                    (void*) compressedBuffer, maxCompressedBufferSize,
                    (void*) buffer, bufferSize,
                    LZMA2_COMPRESSION_LEVEL, 6);

    const char header[5] = "CDXF";

    std::fstream outStream = std::fstream(archiveFile.string(), std::ios::out | std::ios::binary);

    outStream.write(header, 5 * sizeof(char));
    outStream.write((char*) &bufferSize, sizeof(size_t));
    outStream.write((char*) &compressedBufferSize, sizeof(size_t));
    outStream.write(compressedBuffer, compressedBufferSize);

    outStream.close();

    delete[] compressedBuffer;
}
