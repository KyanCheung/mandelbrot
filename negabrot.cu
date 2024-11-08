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
void negabrot(float c_x, float c_y, int area, int max_iter, float d,
               float& R, float& G, float& B, float gradient) {
    // Iterated value of z and the square of its absolute value
    float z_x = c_x;
    float z_y = c_y;
    float z_abs = hypotf(z_x, z_y);
    float theta = atan2f(z_y, z_x);

    int iter = 1;
    while (iter < max_iter && z_abs < (1 << 8)) {
        ++iter;
        
        // Compute z^d
        z_abs = powf(z_abs, d);
        theta *= d;
        // Back into cartesian
        z_x = z_abs * cosf(theta) + c_x;
        z_y = z_abs * sinf(theta) + c_y;

        // Convert back into polar form
        z_abs = hypotf(z_x, z_y);
        theta = atan2f(z_y, z_x);
    }

    float R_sub = 0;
    float G_sub = 0;
    float B_sub = 0;

    // Non-black colour if not in negabrot set
    if (iter != max_iter) {
        // Smoothened value for number of iterations
        float iter_smooth = logf(z_abs) / iter;

        // This is then converted into hue - formula somewhat based on Wikipedia's
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

// Kernel that runs negabrot on subpixel and averages RGB values.
// This subsampling results in an anti-aliasing effect.
__global__
void negabrot_aa(int size_x, int size_y, int grid_size, int max_iter, float d,
                  float x_bounds[2], float y_bounds[2], float gradient, unsigned char *colours) {
    int index_x = blockIdx.x * blockDim.x + threadIdx.x;
    int index_y = blockIdx.y * blockDim.y + threadIdx.y;
    // Position in colours
    int pos = index_y * size_x + index_x;
    int area = size_x * size_y;

    // Variation of c_x/c_y in neighbouring subpixels
    float x_stride = float(x_bounds[1] - x_bounds[0]) / (grid_size * size_x - 1);
    float y_stride = float(y_bounds[1] - y_bounds[0]) / (grid_size * size_y - 1);

    float R = 0;
    float G = 0;
    float B = 0;
    float c_x;
    float c_y;
    // Running negabrot in each subpixel
    for (int grid_x = 0; grid_x < grid_size; ++grid_x) {
        for (int grid_y = 0; grid_y < grid_size; ++grid_y) {
            c_x = (grid_size * index_x + grid_x) * x_stride + x_bounds[0];
            c_y = (grid_size * index_y + grid_y) * y_stride + y_bounds[0];
            negabrot(c_x, c_y, area, max_iter, d, R, G, B, gradient);
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
    int size_x = 2000;
    int size_y = 2000;
    int area = size_x * size_y;
    // Max number of iterations before giving up
    int max_iter = 1000;
    // Boundaries of image on Cartesian plane
    float x_bounds[2] = {-2, 2};
    float y_bounds[2] = {-2, 2};
    
    // How much colour spatially varies
    float gradient = 15;
    // Grid size of supersampling
    int grid_size = 4;

    // Array of colours
    unsigned char* colours;
    cudaMallocManaged(&colours, 3 * area * sizeof(unsigned char));

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(size_x / dimBlock.x, size_y / dimBlock.y);

    // Variables to contain info about PNG
    png::image<png::rgb_pixel> image(size_x, size_y);
    int x;
    int y;
    
    float d = -1.0f;
    for (int n = 0; n < 200; n ++) {
        std::cout << "Generating picture " << n << "..." << std::endl;
        std::cout << "d=" << d << std::endl;
        std::cout << std::endl;
        negabrot_aa<<<dimGrid, dimBlock>>>(size_x, size_y, grid_size, max_iter, d, x_bounds, y_bounds, gradient, colours);

        // Wait for GPU to finish before accessing on host
        cudaDeviceSynchronize();

        // Write to PNG
        for (int i = 0; i < area; ++i) {
            y = i / size_x;
            x = i % size_x;
            image[y][x] = png::rgb_pixel(colours[i], colours[i + area], colours[i + 2 * area]);
        }
        image.write(std::format("images/negabrot-{}.png", n));

        // Increment d
        d -= 0.02f;
    }

    return 0;
}