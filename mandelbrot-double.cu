#include <iomanip>
#include <cmath>
#include <fstream>
#include <png++/image.hpp>
#include <png++/rgb_pixel.hpp>
#include <cstdio>
#include <format>

// Thread block size
#define BLOCK_SIZE 16

// Kernel that calculates what colour to give to each subpixel
__device__
void mandelbrot(double c_x, double c_y, int area, int max_iter,
                float& R, float& G, float& B, double gradient) {

    // Check if c is in cardioid
    double q = (c_x - .25) * (c_x - .25) + c_y * c_y;
    if (4 * (q * (q + c_x - .25)) < c_y * c_y) {
        return;
    }
    // Check if c is in period-2 bulb
    if (16 * ((c_x + 1) * (c_x + 1) + c_y * c_y) < 1) {
        return;
    }
    // Iterated value of z and the square of its absolute value
    double z_x = 0;
    double z_y = 0;
    double z_abs = 0;
    double z_xtemp = 0;
    double z_ytemp = 0;

    int iter = 0;
    while (iter < max_iter && z_abs <= (1 << 16)) {
        ++iter;
        
        // z -> z^2 + c
        z_xtemp = (z_x - z_y) * (z_x + z_y) + c_x;
        z_ytemp = 2*z_x*z_y + c_y;
        z_x = z_xtemp;
        z_y = z_ytemp;

        z_abs = z_x * z_x + z_y * z_y;
    }

    float R_sub = 0;
    float G_sub = 0;
    float B_sub = 0;

    // Non-black colour if not in Mandelbrot set
    if (iter != max_iter) {
        // Smoothened value for number of iterations
        float iter_smooth = iter - log2f(logf(z_abs)) + 1;

        // This is then converted into hue -ormula somewhat based on Wikipedia's
        float H = fmodf(iter_smooth * sqrt(iter_smooth) * gradient, 360);

        // From hue we convert to RGB (with full saturation and value)
        // Only 2 of 3 colours have nonzero value as S=100
        if (H < 60) {
            R_sub = 255;
            G_sub = 255 * H / 60;
        } else if (H < 120) {
            R_sub = 255 * (120 - H) / 60;
            G_sub = 255;
        } else if (H < 180) {
            G_sub = 255;
            B_sub = 255 * (H - 120) / 60;
        } else if (H < 240) {
            G_sub = 255 * (240 - H) / 60;
            B_sub = 255;
        } else if (H < 300) {
            B_sub = 255;
            R_sub = 255 * (H - 240) / 60;
        } else {
            B_sub = 255 * (360 - H) / 60;
            R_sub = 255;
        }

        R += R_sub;
        G += G_sub;
        B += B_sub;
    }
}

// Kernel that runs mandelbrot on subpixel and averages RGB values.
// This subsampling results in an anti-aliasing effect.
__global__
void mandelbrot_aa(int size_x, int size_y, int grid_size, int max_iter, 
                   double x_bounds[2], double y_bounds[2], float gradient, char *colours) {
    int index_x = blockIdx.x * blockDim.x + threadIdx.x;
    int index_y = blockIdx.y * blockDim.y + threadIdx.y;
    // Position in colours
    int pos = index_y * size_x + index_x;
    int area = size_x * size_y;

    // Variation of c_x/c_y in neighbouring subpixels
    double x_stride = double(x_bounds[1] - x_bounds[0]) / (grid_size * size_x - 1);
    double y_stride = double(y_bounds[1] - y_bounds[0]) / (grid_size * size_y - 1);

    float R = 0;
    float G = 0;
    float B = 0;
    double c_x;
    double c_y;
    // Running mandelbrot in each subpixel
    for (int grid_x = 0; grid_x < grid_size; ++grid_x) {
        for (int grid_y = 0; grid_y < grid_size; ++grid_y) {
            c_x = (grid_size * index_x + grid_x) * x_stride + x_bounds[0];
            c_y = (grid_size * index_y + grid_y) * y_stride + y_bounds[0];
            mandelbrot(c_x, c_y, area, max_iter, R, G, B, gradient);
        }
    }

    // Normalise colours by dividing by no. of subpixels
    R /= grid_size * grid_size;
    G /= grid_size * grid_size;
    B /= grid_size * grid_size;

    // Write colours
    // Colours stored like this to maximise thread coalescing
    colours[pos] = R;
    colours[pos + area] = G;
    colours[pos + area * 2] = B;
}

int main() {
    // Number of pixels in image
    int size_x = 3840;
    int size_y = 2160;
    int area = size_x * size_y;
    // Max number of iterations before giving up
    int max_iter = 10000;
    // Boundaries of image on Cartesian plane
    double x_bounds[2] = {-3.2, 1.6};
    double y_bounds[2] = {1.35, -1.35};
    double x_centre = -0.8212007965493;
    double y_centre = -0.200572441411;
    // How much colour spatially varies
    double gradient = 1.7;
    // Grid size of supersampling
    int grid_size = 4;
    // Rate of zooming in
    double zoom = 1.03;

    // Array of colours
    char* colours;
    cudaMallocManaged(&colours, 3 * area * sizeof(char));

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(size_x / dimBlock.x, size_y / dimBlock.y);

    // Variables to contain info about PNG
    png::image<png::rgb_pixel> image(size_x, size_y);
    int x;
    int y;
    
    for (int n = 0; n < 1000; n ++) {
        std::cout << "Generating picture " << n << "..." << std::endl;
        std::cout << "x-bounds: " << std::setprecision(16) << x_bounds[0] << ", " << std::setprecision(16) << x_bounds[1] << std::endl;
        std::cout << "y-bounds: " << std::setprecision(16) << y_bounds[1] << ", " << std::setprecision(16) << y_bounds[0] << std::endl;
        std::cout << std::endl;

        mandelbrot_aa<<<dimGrid, dimBlock>>>(
            size_x, size_y, grid_size, max_iter, x_bounds, y_bounds, gradient, colours
        );
        // Wait for GPU to finish before accessing on host
        cudaDeviceSynchronize();

        // Write to PNG
        for (int i = 0; i < area; ++i) {
            y = i / size_x;
            x = i % size_x;
            image[y][x] = png::rgb_pixel(colours[i], colours[i + area], colours[i + 2 * area]);
        }
        image.write(std::format("images/mandelbrot-{}.png", n));

        // Set new boundaries for image
        x_bounds[0] = (x_bounds[0] - x_centre) / zoom + x_centre;
        x_bounds[1] = (x_bounds[1] - x_centre) / zoom + x_centre;
        y_bounds[0] = (y_bounds[0] - y_centre) / zoom + y_centre;
        y_bounds[1] = (y_bounds[1] - y_centre) / zoom + y_centre;
    }

    return 0;
}