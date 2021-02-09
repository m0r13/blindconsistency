#include "flowIO.h"

#include <cstdlib>
#include <stdexcept>

#define TAG_FLOAT 202021.25  // check for this when READING the file
#define TAG_STRING "PIEH"    // use this when WRITING the file

// read a flow file into 2-band image
void ReadFlowFile(std::vector<float>& flow, std::string filename)
{
    FILE* stream = fopen(filename.c_str(), "rb");
    if (stream == 0) {
        throw std::exception((std::string("ReadFlowFile: could not open ") + filename).c_str());
    }

    int width, height;
    float tag;

    if ((int) fread(&tag, sizeof(float), 1, stream) != 1 ||
        (int) fread(&width, sizeof(int), 1, stream) != 1 ||
        (int) fread(&height, sizeof(int), 1, stream) != 1) {
        throw std::exception((std::string("ReadFlowFile: problem reading file ") + filename).c_str());
    }

    if (tag != TAG_FLOAT) { // simple test for correct endian-ness
        throw std::exception((std::string("ReadFlowFile: wrong tag (possibly due to big-endian machine?) ") + filename).c_str());
    }

    // another sanity check to see that integers were read correctly (99999 should do the trick...)
    if (width < 1 || width > 99999) {
        throw std::exception((std::string("ReadFlowFile: illegal width ") + filename).c_str());
    }

    if (height < 1 || height > 99999) {
        throw std::exception((std::string("ReadFlowFile: illegal height ") + filename).c_str());
    }

    int nBands = 2;
    flow.resize((size_t) height * width * nBands);

    //printf("reading %d x %d x 2 = %d floats\n", width, height, width*height*2);
    int n = nBands * width;
    for (int y = 0; y < height; y++) {
        //float* ptr = &img.Pixel(0, y, 0);
        float* ptr = &flow[(size_t) y * width * 2];
        if ((int) fread(ptr, sizeof(float), n, stream) != n) {
            throw std::exception((std::string("ReadFlowFile: file is too short ") + filename).c_str());
        }
    }

    if (fgetc(stream) != EOF) {
        throw std::exception((std::string("ReadFlowFile: file is too long ") + filename).c_str());
    }

    fclose(stream);
}

// write a 2-band image into flow file 
void WriteFlowFile(const std::vector<float>& flow, int W, int H, std::string filename)
{
    int width = W, height = H, nBands = 2;

    FILE* stream = fopen(filename.c_str(), "wb");
    if (stream == 0) {
        throw std::exception((std::string("ReadFlowFile: could not open ") + filename).c_str());
    }

    // write the header
    fprintf(stream, TAG_STRING);
    if ((int) fwrite(&width, sizeof(int), 1, stream) != 1 ||
        (int) fwrite(&height, sizeof(int), 1, stream) != 1) {
        throw std::exception((std::string("ReadFlowFile: can't write header to ") + filename).c_str());
    }

    // write the rows
    int n = nBands * width;
    for (int y = 0; y < height; y++) {
        const float* ptr = &flow[(size_t) y * width * 2];
        if ((int) fwrite(ptr, sizeof(float), n, stream) != n) {
            throw std::exception((std::string("ReadFlowFile: can't write data to ") + filename).c_str());
        }
    }

    fclose(stream);
}