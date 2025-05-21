import numpy as np
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms


def generate_random_data(height, width, count=1):

    shape = (height, width)
    images = []
    masks  = []

    for _ in range(count):

        triangle_location = get_random_location(height, width)
        circle_location1  = get_random_location(height, width, zoom=0.7)
        circle_location2  = get_random_location(height, width, zoom=0.5)
        mesh_location     = get_random_location(height, width)
        square_location   = get_random_location(height, width, zoom=0.8)
        plus_location     = get_random_location(height, width, zoom=1.2)

        # Create input image
        img = np.zeros(shape, dtype=bool)
        img = add_triangle(img, *triangle_location)
        img = add_circle(img, *circle_location1)
        img = add_circle(img, *circle_location2, fill=True)
        img = add_mesh_square(img, *mesh_location)
        img = add_filled_square(img, *square_location)
        img = add_plus(img, *plus_location)
        img = np.reshape(img, (height, width, 1)).astype(np.float32)

        # Create target masks
        mask = np.asarray([
            add_triangle(np.zeros(shape, dtype=bool), *triangle_location),
            add_circle(np.zeros(shape, dtype=bool), *circle_location1),
            add_circle(np.zeros(shape, dtype=bool), *circle_location2, fill=True),
            add_mesh_square(np.zeros(shape, dtype=bool), *mesh_location),
            add_filled_square(np.zeros(shape, dtype=bool), *square_location),
            add_plus(np.zeros(shape, dtype=bool), *plus_location)
        ])
        mask = np.transpose(mask, (1, 2, 0))

        images.append(img)
        masks.append(mask)

    images = np.stack(images)
    masks = np.stack(masks)

    images = (images*255).astype(np.uint8)

    return images, masks


def add_square(arr, x, y, size):
    s = int(size / 2)
    arr[x-s,y-s:y+s] = True
    arr[x+s,y-s:y+s] = True
    arr[x-s:x+s,y-s] = True
    arr[x-s:x+s,y+s] = True

    return arr

def add_filled_square(arr, x, y, size):
    s = int(size / 2)

    xx, yy = np.mgrid[:arr.shape[0], :arr.shape[1]]

    return np.logical_or(arr, logical_and([xx > x - s, xx < x + s, yy > y - s, yy < y + s]))

def logical_and(arrays):
    new_array = np.ones(arrays[0].shape, dtype=bool)
    for a in arrays:
        new_array = np.logical_and(new_array, a)

    return new_array

def add_mesh_square(arr, x, y, size):
    s = int(size / 2)

    xx, yy = np.mgrid[:arr.shape[0], :arr.shape[1]]

    return np.logical_or(arr, logical_and([xx > x - s, xx < x + s, xx % 2 == 1, yy > y - s, yy < y + s, yy % 2 == 1]))

def add_triangle(arr, x, y, size):
    s = int(size / 2)

    triangle = np.tril(np.ones((size, size), dtype=bool))

    arr[x-s:x-s+triangle.shape[0],y-s:y-s+triangle.shape[1]] = triangle

    return arr

def add_circle(arr, x, y, size, fill=False):
    xx, yy = np.mgrid[:arr.shape[0], :arr.shape[1]]
    circle = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
    new_arr = np.logical_or(arr, np.logical_and(circle < size, circle >= size * 0.7 if not fill else True))

    return new_arr

def add_plus(arr, x, y, size):
    s = int(size / 2)
    arr[x-1:x+1,y-s:y+s] = True
    arr[x-s:x+s,y-1:y+1] = True

    return arr

def get_random_location(width, height, zoom=1.0):
    x = int(width * random.uniform(0.1, 0.9))
    y = int(height * random.uniform(0.1, 0.9))

    size = int(min(width, height) * random.uniform(0.06, 0.12) * zoom)

    return (x, y, size)

class ShapesDataset(Dataset):
    def __init__(self, count):
        self.input_images, self.target_masks = generate_random_data(192, 192, count=count)        
        self.transform = transforms.ToTensor()
    
    def __len__(self):
        return len(self.input_images)
    
    def __getitem__(self, idx):        
        image = self.transform(self.input_images[idx])
        mask = self.transform(self.target_masks[idx]).float()
        
        return image, mask