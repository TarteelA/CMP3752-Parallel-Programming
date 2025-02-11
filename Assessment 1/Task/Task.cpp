//Coded by: Tarteel Alkaraan (25847208)
//Last Updated: 29/03/2024
//This is an edited and extended version of Tutorial 3 from (Wingate, 2024), (Jameson, 2023) & (Gregorio, 2020)
#include <iostream>
#include <vector>
#include "Utils.h"
#include "CImg.h"
#include <CL/cl.h>

using namespace cimg_library;

void print_help() {
    std::cerr << "Usage of App:" << std::endl;
    std::cerr << "-p : Platform choosing" << std::endl;
    std::cerr << "-d : Device choosing" << std::endl;
    std::cerr << "-l : Show all devices and platforms" << std::endl;
    std::cerr << "-f : File picture input" << std::endl;
    std::cerr << "-h : Show message" << std::endl;
}

void histogram_calculation() {

}

int main(int argc, char** argv) {
    //Section 1 Command line handle choices 
    int platform_id = 0;
    int device_id = 0;
    string path_picture = "test.ppm";

    for (int i = 1; i < argc; i++) {
        if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
        else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
        else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
        else if ((strcmp(argv[i], "-f") == 0) && (i < (argc - 1))) { path_picture = argv[++i]; }
        else if (strcmp(argv[i], "-h") == 0) { print_help(); return 0; }
    }
    cimg::exception_mode(0);

    //Get input from user number of bins
    string userCommand;
    int num_bin = 0;
    std::cout << "\nInput number of bin from 0 to 256" << "\n";
    //Will continue until user enters a number
    while (true)
    {
        //Receives users input
        getline(std::cin, userCommand);

        //Makes sure users input isn't null
        if (userCommand == "") { std::cout << "Input number please." << "\n"; continue; }

        //Try to change users input to an int
        try { num_bin = std::stoi(userCommand); }
        catch (...) { std::cout << "Input integer please." << "\n"; continue; }
        
        //Makes sure users input is within specified range
        if (num_bin >= 0 && num_bin <= 256) { break; }
        else { std::cout << "Input number from 0 to 256 please." << "\n"; continue; }
    }

    //Identify exceptions possible
    try {
        //Read picture from file path
        CImg<unsigned char> input_picture(path_picture.c_str());
        CImgDisplay disp_input(input_picture, "input");

        //Implementing 3x3 mask filter convolution averaging
        std::vector<float> mask_convolution = { 1.f / 9, 1.f / 9, 1.f / 9, 1.f / 9, 1.f / 9, 1.f / 9, 1.f / 9, 1.f / 9, 1.f / 9 };

        //Section 2 Operations host
        //Choose device computings
        cl::Context context = GetContext(platform_id, device_id);

        //Show device choosen
        std::cout << "\nDevice running " << GetPlatformName(platform_id) << "," << GetDeviceName(platform_id, device_id) << std::endl;

        //Queue making to push device commands
        cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);

        //Import and construct code device
        cl::Program::Sources sources;

        AddSources(sources, "kernels.cl");

        cl::Program program(context, sources);

        //Debug and construct code kernel
        try {
            program.build();
        }
        catch (const cl::Error& err) {
            std::cout << "Status Construct:" << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
            std::cout << "Choices Construct:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
            std::cout << "Log Construct:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
            throw err;
        }

        CImg<unsigned char> Cb;
        CImg<unsigned char> Cr;
        bool isColour = input_picture.spectrum() == 3;
        if (isColour) {
            std::cout << "\nColour Picture. " << std::endl;
            //Change picture to YCbCr to get channel intensity
            CImg<unsigned char> picture_temp = input_picture.get_RGBtoYCbCr();
            input_picture = picture_temp.get_channel(0);
            Cb = picture_temp.get_channel(1);
            Cr = picture_temp.get_channel(2);
        }
        else
        {
            std::cout << "Greyscale Picture." << std::endl;
        }
        
        //Section 3 Distribution Memory
        //Input Host
        //Makes and initialises size of vector num_bin with 0
        std::vector<int>h(num_bin);
        //Compute in bytes the buffer of histogram size
        size_t sizehist = h.size() * sizeof(int);

        //Make Buffer Devices for picture input, picture output, LUT, hist, and hist cumulative
        cl::Buffer dev_input_picture(context, CL_MEM_READ_ONLY, input_picture.size());
        cl::Buffer dev_output_picture(context, CL_MEM_READ_WRITE, input_picture.size());
        cl::Buffer dev_output_hist(context, CL_MEM_READ_WRITE, sizehist);
        cl::Buffer dev_output_hist_cumulative(context, CL_MEM_READ_WRITE, sizehist);
        cl::Buffer dev_output_LUT(context, CL_MEM_READ_WRITE, sizehist);

        //Section 4 Operation Devices
        //Pictures copied to memory device
        queue.enqueueWriteBuffer(dev_input_picture, CL_TRUE, 0, input_picture.size(), &input_picture.data()[0]);

        //Execute and Setup kernel
        cl::Kernel kernel_hist = cl::Kernel(program, "hist");
        kernel_hist.setArg(0, dev_input_picture);
        kernel_hist.setArg(1, dev_output_hist);
        cl::Event prof_event;

        queue.enqueueNDRangeKernel(kernel_hist, cl::NullRange, cl::NDRange(input_picture.size()), cl::NullRange, NULL, &prof_event);
        queue.enqueueReadBuffer(dev_output_hist, CL_TRUE, 0, sizehist, &h[0]);

        std::cout << "\nHistogram values:\n" << std::endl;
        for (int x : h) {
            std::cout << x << " ";
        }

        std::vector<int> Ch(num_bin);

        queue.enqueueFillBuffer(dev_output_hist_cumulative, 0, 0, sizehist);

        cl::Kernel kernel_hist_cum = cl::Kernel(program, "hist_cum");
        kernel_hist_cum.setArg(0, dev_output_hist);
        kernel_hist_cum.setArg(1, dev_output_hist_cumulative);
        cl::Event prof_event2;

        queue.enqueueNDRangeKernel(kernel_hist_cum, cl::NullRange, cl::NDRange(sizehist), cl::NullRange, NULL, &prof_event2);
        queue.enqueueReadBuffer(dev_output_hist_cumulative, CL_TRUE, 0, sizehist, &Ch[0]);

        std::cout << "\n\nHistogram Cumulative values:\n" << std::endl;
        for (int x : Ch) {
            std::cout << x << " ";
        }

        std::vector<int> LUT(num_bin);

        queue.enqueueFillBuffer(dev_output_LUT, 0, 0, sizehist);

        cl::Kernel kernel_LUT = cl::Kernel(program, "LUT");
        kernel_LUT.setArg(0, dev_output_hist_cumulative);
        kernel_LUT.setArg(1, dev_output_LUT);
        cl::Event prof_event3;

        queue.enqueueNDRangeKernel(kernel_LUT, cl::NullRange, cl::NDRange(sizehist), cl::NullRange, NULL, &prof_event3);
        queue.enqueueReadBuffer(dev_output_LUT, CL_TRUE, 0, sizehist, &LUT[0]);

        cl::Kernel kernel_ReProject = cl::Kernel(program, "ReProject");
        kernel_ReProject.setArg(0, dev_input_picture);
        kernel_ReProject.setArg(1, dev_output_LUT);
        kernel_ReProject.setArg(2, dev_output_picture);
        cl::Event prof_event4;

        vector<unsigned char> buffer_output(input_picture.size());
        queue.enqueueNDRangeKernel(kernel_ReProject, cl::NullRange, cl::NDRange(input_picture.size()), cl::NullRange, NULL, &prof_event4);
        queue.enqueueReadBuffer(dev_output_picture, CL_TRUE, 0, buffer_output.size(), &buffer_output.data()[0]);

        int time_hist = prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
        std::cout << "\n\nHistogram execution time kernel: " << time_hist << std::endl;
        std::cout << "\nMemory transfer histogram: " << GetFullProfilingInfo(prof_event, ProfilingResolution::PROF_US) << std::endl << std::endl;;

        int hist_cum_time = prof_event2.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event2.getProfilingInfo<CL_PROFILING_COMMAND_START>();
        std::cout << "Histogram Cumulative execution time kernel: " << hist_cum_time << std::endl;
        std::cout << "\nMemory transfer histogram cumulative: " << GetFullProfilingInfo(prof_event2, ProfilingResolution::PROF_US) << std::endl << std::endl;;

        int time_lut = prof_event3.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event3.getProfilingInfo<CL_PROFILING_COMMAND_START>();
        std::cout << "LUT: " << time_lut << std::endl;
        std::cout << "\nMemory transfer LUT: " << GetFullProfilingInfo(prof_event3, ProfilingResolution::PROF_US) << std::endl << std::endl;;

        int time_ReProject = prof_event4.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event4.getProfilingInfo<CL_PROFILING_COMMAND_START>();
        std::cout << "ReProject execution time: " << time_ReProject << std::endl;
        std::cout << "\nMemory tranfer ReProject: " << GetFullProfilingInfo(prof_event4, ProfilingResolution::PROF_US) << endl;

        std::cout << "\nProgram execution time sum: " << time_hist + hist_cum_time + time_lut + time_ReProject << std::endl;

        //Rebuild our last output picture from normalised histogram values
        CImg<unsigned char>picture_output(buffer_output.data(), input_picture.width(), input_picture.height(), input_picture.depth(), input_picture.spectrum());

        if (isColour) {
            //The picture was rgb
            CImg<unsigned char> YCbCrPic(picture_output.width(), picture_output.height(), 1, 3);
            for (int x = 0; x < picture_output.width(); x++) {
                for (int y = 0; y < picture_output.height(); y++) {
                    YCbCrPic(x, y, 0) = picture_output(x, y);
                    YCbCrPic(x, y, 1) = Cb(x, y);
                    YCbCrPic(x, y, 2) = Cr(x, y);
                }
            }

            //Change back to rgb
            picture_output = YCbCrPic.get_YCbCrtoRGB();
        }

        CImgDisplay output_display(picture_output, "output");
        while (!disp_input.is_closed() && !output_display.is_closed()
            && !disp_input.is_keyESC() && !output_display.is_keyESC()) {
            disp_input.wait(1);
            output_display.wait(1);
        }
    }
    catch (const cl::Error& err) {
        std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
    }
    catch (CImgException& err) {
        std::cerr << "ERROR: " << err.what() << std::endl;
    }
    return 0;
}