
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt

def conv_single_step(a_slice_prev, W, b):
    """
    Apply one filter defined by parameters W on a single slice (a_slice_prev) of the output activation
    of the previous layer.

    Arguments:
    a_slice_prev -- slice of input data of shape (f, f, n_C_prev)
    W -- Weight parameters contained in a window - matrix of shape (f, f, n_C_prev)
    b -- Bias parameters contained in a window - matrix of shape (1, 1, 1)

    Returns:
    Z -- a scalar value, result of convolving the sliding window (W, b) on a slice x of the input data
    """

    ### START CODE HERE ### (≈ 2 lines of code)
    # Element-wise product between a_slice_prev and W. Do not add the bias yet.
    s = a_slice_prev * W
    # Sum over all entries of the volume s.
    Z = np.sum(s)
    # Add bias b to Z. Cast b to a float() so that Z results in a scalar value.
    Z = np.float(np.add(b, Z))
    ### END CODE HERE ###

    return Z



def convolve2d(image, kernel, padding=False, striding=1):
    # This function which takes an image and a kernel
    # and returns the convolution of them
    # Args:
    #   image: a numpy array of size [image_height, image_width].
    #   kernel: a numpy array of size [kernel_height, kernel_width].
    # Returns:
    #   a numpy array of size [image_height, image_width] (convolution output).

    assert kernel.shape[0] == kernel.shape[1], "kernel must be square"
    assert striding != 0, "striding cannot be zero"

    # The kernel is flipped so that we are not performing a "correlation" operation
    kernel = np.flipud(np.fliplr(kernel))

    kernel_h = kernel.shape[0]
    kernel_w = kernel.shape[1]

    h = kernel_h // 2
    w = kernel_w // 2

    image_h = image.shape[0]
    image_w = image.shape[1]

    # if padding turned on (to fix border effect) then set for "same" padding
    if padding:
        pad = (kernel_h - 1) // 2
    else:
        pad = 0

    new_height = int(((image_h + 2*pad - kernel_h) / striding) + 1)
    new_width = int(((image_w + 2*pad - kernel_w) / striding) + 1)
    image_out = np.zeros(new_height, new_width)

    # Add padding to the input image
    image_padded = np.pad(image, ((0,0), (pad, pad), (pad, pad), (0,0)), 'constant', constant_values = (0,0))

    for x in range(h, image_h - h):  # Loop over every pixel of the image
        for y in range(w, image_w - w):
            sum = 0

            for m in range(kernel_h):
                for n in range(kernel_w):
                    sum += kernel[m][n] * image_padded[x-h+m][y-w+n]

            image_out[x,y] = sum

    return image_out

img = misc.ascent()
plt.grid(False)
plt.gray()
plt.axis('off')
plt.imshow(img)
plt.show()


# This filter detects edges nicely
# It creates a convolution that only passes through sharp edges and straight
# lines.

#Experiment with different values for fun effects.

filter_edge = [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
image_sharpen = convolve2d(img, filter_edge)
plt.imshow(image_sharpen, cmap=plt.cm.gray)
plt.axis('off')
plt.show()



# A couple more filters to try for fun!
filter = [ [-1, -2, -1], [0, 0, 0], [1, 2, 1]]


filter = [ [0, 1, 1, 0], [1, 3, 3, 1], [-1, -3, -3, -1], [0, -1, -1, 0]]
weight = 1

#filter = [ [-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]

# If all the digits in the filter don't add up to 0 or 1, you
# should probably do a weight to get it to do so
# so, for example, if your weights are 1,1,1 1,2,1 1,1,1
# They add up to 10, so you would set a weight of .1 if you want to normalize them


i_transformed = np.copy(i)
size_x = i_transformed.shape[0]
size_y = i_transformed.shape[1]
print(size_x, size_y)

weight = 1

for x in range(2,size_x-2):
  for y in range(2,size_y-2):
      convolution = 0.0
      convolution = convolution + (i[x - 2, y-2] * filter[0][0])
      convolution = convolution + (i[x - 1, y-2] * filter[0][1])
      convolution = convolution + (i[x, y-2] * filter[0][2])
      convolution = convolution + (i[x + 1, y-2] * filter[0][3])

      convolution = convolution + (i[x-1, y] * filter[1][0])
      convolution = convolution + (i[x, y] * filter[1][1])
      convolution = convolution + (i[x+1, y] * filter[1][2])
      convolution = convolution + (i[x + 1, y] * filter[1][3])

      convolution = convolution + (i[x-1, y+1] * filter[2][0])
      convolution = convolution + (i[x, y+1] * filter[2][1])
      convolution = convolution + (i[x+1, y+1] * filter[2][2])
      convolution = convolution + (i[x + 1, y + 1] * filter[2][3])

      convolution = convolution + (i[x-1, y+1] * filter[3][0])
      convolution = convolution + (i[x, y+1] * filter[3][1])
      convolution = convolution + (i[x+1, y+1] * filter[3][2])
      convolution = convolution + (i[x + 1, y + 1] * filter[3][3])



      convolution = convolution * weight
      if(convolution<0):
        convolution=0
      if(convolution>255):
        convolution=255
      i_transformed[x, y] = convolution

# Plot the image. Note the size of the axes -- they are 512 by 512
plt.gray()
plt.grid(False)
plt.imshow(i_transformed)
#plt.axis('off')
plt.show()

plt.imshow(image_sharpen, cmap=plt.cm.gray)
plt.axis('off')
plt.show()