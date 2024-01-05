from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import numpy as np
import os
import cv2
from zipfile import ZipFile

def flip_horizontal_image(image):
    img = Image.fromarray(image)
    return np.array(ImageOps.mirror(img))

def flip_vertical_image(image):
    img = Image.fromarray(image)
    return np.array(ImageOps.flip(img))

def rotate_image(image, angle=90):
    img = Image.fromarray(image)
    return np.array(img.rotate(angle))

def random_rotation(image):
    angle = np.random.randint(1, 360)
    img = Image.fromarray(image)
    return np.array(img.rotate(angle))

def random_shear(image, shear_factor=0.2):
    img = Image.fromarray(image)
    shear_factor = np.random.uniform(-shear_factor, shear_factor)
    return np.array(img.transform(img.size, Image.AFFINE, (1, shear_factor, 0, 0, 1, 0), resample=Image.BICUBIC))

def random_crop(image, crop_percent=0.2):
    img = Image.fromarray(image)
    width, height = img.size
    left = np.random.randint(0, int(crop_percent * width))
    upper = np.random.randint(0, int(crop_percent * height))
    right = width - np.random.randint(0, int(crop_percent * width))
    lower = height - np.random.randint(0, int(crop_percent * height))
    return np.array(img.crop((left, upper, right, lower)))

def apply_blur(image, radius=2):
    img = Image.fromarray(image)
    return np.array(img.filter(ImageFilter.GaussianBlur(radius=radius)))

def apply_exposure(image, exposure_factor=1.5):
    img = Image.fromarray(image)
    enhancer = ImageEnhance.Brightness(img)
    return np.array(enhancer.enhance(exposure_factor))

def apply_random_noise(image, noise_factor=20):
    img = Image.fromarray(image)
    
    # Get the number of channels in the image
    num_channels = len(img.getbands())
    
    # Generate noise for each channel
    noise = [np.random.normal(scale=noise_factor, size=img.size[::-1]) for _ in range(num_channels)]
    
    # Clip noise values to be within the valid range for image dtype
    noise = [np.clip(channel, 0, 255) for channel in noise]
    
    # Combine noise channels
    noisy_img = np.stack(noise, axis=-1).astype(np.uint8)
    
    # Add noise to the image
    noisy_img = np.clip(np.array(img) + noisy_img, 0, 255)
    
    return noisy_img


def cutout(image, cutout_size=100):
    img = Image.fromarray(image)
    width, height = img.size
    left = np.random.randint(0, width - cutout_size)
    upper = np.random.randint(0, height - cutout_size)
    right = left + cutout_size
    lower = upper + cutout_size
    img.paste((0, 0, 0), (left, upper, right, lower))
    return np.array(img)

def mosaic(image, mosaic_size=5):
    img = Image.fromarray(image)
    width, height = img.size
    for _ in range(mosaic_size):
        left = np.random.randint(0, width)
        upper = np.random.randint(0, height)
        right = np.random.randint(left, width)
        lower = np.random.randint(upper, height)
        img.crop((left, upper, right, lower)).paste(img.crop((left, upper, right, lower)).resize((1, 1)))
    return np.array(img)

def color_jitter(image, brightness_factor=0.5, contrast_factor=0.5, saturation_factor=0.5, hue_factor=0.5):
    img = Image.fromarray(image)
    
    # Adjust brightness
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(1 + brightness_factor * np.random.uniform(-1, 1))
    
    # Adjust contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1 + contrast_factor * np.random.uniform(-1, 1))
    
    # Adjust saturation
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(1 + saturation_factor * np.random.uniform(-1, 1))
    
    # Adjust hue
    img = img.convert('HSV')
    img = np.array(img)
    img[:, :, 0] = (img[:, :, 0] + hue_factor * 255) % 256
    img = Image.fromarray(img, 'HSV').convert('RGB')
    
    return np.array(img)

def rotate_with_bounding_box(image, angle=30):
    img = Image.fromarray(image)
    rotated_img = img.rotate(angle, resample=Image.BICUBIC, center=(img.width // 2, img.height // 2))
    return np.array(rotated_img)

def clahe_equalization(image, clip_limit=2.0, grid_size=(8, 8)):
    img = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    img[:, :, 0] = clahe.apply(img[:, :, 0])
    img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
    return img

def random_zoom(image, zoom_range=(0.8, 1.2)):
    img = Image.fromarray(image)
    zoom_factor = np.random.uniform(*zoom_range)
    new_size = tuple(int(dim * zoom_factor) for dim in img.size)
    img = img.resize(new_size, Image.BICUBIC)
    return np.array(img)

def channel_shuffle(image):
    img = Image.fromarray(image)

    # Convert to RGB mode if not already
    if img.mode != 'RGB':
        img = img.convert('RGB')

    channels = list(img.split())
    np.random.shuffle(channels)
    shuffled_img = Image.merge(img.mode, channels)
    return np.array(shuffled_img)


def histogram_equalization(image):
    img = Image.fromarray(image)

    # Convert to RGB mode if not already
    if img.mode != 'RGB':
        img = img.convert('RGB')

    img = ImageOps.equalize(img)
    return np.array(img)

def get_augmented_images(input_dir ,output_dir):
    for filename in os.listdir(input_dir):
        original_image = np.array(Image.open(os.path.join(input_dir, filename)))
        
        flipped_image_horizontal = flip_horizontal_image(original_image)
        flipped_image_vertical = flip_vertical_image(original_image)
        rotated_image = rotate_image(original_image, angle=90)
        randomly_rotated_image = random_rotation(original_image)
        sheared_image = random_shear(original_image, shear_factor=0.2)
        cropped_image = random_crop(original_image, crop_percent=0.2)
        blurred_image = apply_blur(original_image, radius=2)
        exposed_image = apply_exposure(original_image, exposure_factor=1.5)
        noisy_image = apply_random_noise(original_image, noise_factor=20)
        cutout_image = cutout(original_image, cutout_size=100)
        mosaic_image = mosaic(original_image, mosaic_size=5)
        jittered_image = color_jitter(original_image)
        equalized_image = histogram_equalization(original_image)
        shuffled_image = channel_shuffle(original_image)
        zoomed_image = random_zoom(original_image)
        clahe_image = clahe_equalization(original_image)
        rotated_bounded_image = rotate_with_bounding_box(original_image, angle=30)
        
        Image.fromarray(flipped_image_horizontal).save(os.path.join(output_dir, f'flipped_horizontal_{filename}'))
        Image.fromarray(flipped_image_vertical).save(os.path.join(output_dir, f'flipped_vertical_{filename}'))
        Image.fromarray(rotated_image).save(os.path.join(output_dir, f'rotated_{filename}'))
        Image.fromarray(randomly_rotated_image).save(os.path.join(output_dir, f'randomly_rotated_{filename}'))
        Image.fromarray(sheared_image).save(os.path.join(output_dir, f'sheared_{filename}'))
        Image.fromarray(cropped_image).save(os.path.join(output_dir, f'cropped_{filename}'))
        Image.fromarray(blurred_image).save(os.path.join(output_dir, f'blurred_{filename}'))
        Image.fromarray(exposed_image).save(os.path.join(output_dir, f'exposed_{filename}'))
        Image.fromarray(noisy_image).save(os.path.join(output_dir, f'noisy_{filename}'))
        Image.fromarray(cutout_image).save(os.path.join(output_dir, f'cutout_{filename}'))
        Image.fromarray(mosaic_image).save(os.path.join(output_dir, f'mosaic_{filename}'))
        Image.fromarray(jittered_image).save(os.path.join(output_dir, f'jittered_{filename}'))
        Image.fromarray(equalized_image).save(os.path.join(output_dir, f'equalized_{filename}'))
        Image.fromarray(shuffled_image).save(os.path.join(output_dir, f'shuffled_{filename}'))
        Image.fromarray(zoomed_image).save(os.path.join(output_dir, f'zoomed_{filename}'))
        Image.fromarray(clahe_image).save(os.path.join(output_dir, f'clahe_{filename}'))
        Image.fromarray(rotated_bounded_image).save(os.path.join(output_dir, f'rotated_bounded_{filename}'))
        print(f'Done {filename}')
        
# image_augmentation_functions = {
#     'flip_horizontal_image': flip_horizontal_image,
#     'flip_vertical_image': flip_vertical_image,
#     'rotate_image': rotate_image,
#     'random_rotation': random_rotation,
#     'random_shear': random_shear,
#     'random_crop': random_crop,
#     'apply_blur': apply_blur,
#     'apply_exposure': apply_exposure,
#     'apply_random_noise': apply_random_noise,
#     'cutout': cutout,
#     'mosaic': mosaic,
#     'color_jitter': color_jitter,
#     'rotate_with_bounding_box': rotate_with_bounding_box,
#     'clahe_equalization': clahe_equalization,
#     'random_zoom': random_zoom,
#     'channel_shuffle': channel_shuffle,
#     'histogram_equalization': histogram_equalization
# }


# user_choice = ['10','7','9']
# list_of_methods = []
# for i in user:
#     list_of_methods.append(image_augmentation_functions[i])
# print(list_of_methods)

# input_dir = 'image_for_showcase/'
# output_dir = 'augmented_images_for_showcase/'
# os.makedirs(output_dir, exist_ok=True)
# get_augmented_images(input_dir,output_dir)



def get_custom_augmented_images(input_dir,output_dir,user_choice):
    image_augmentation_functions = {
    1: flip_horizontal_image,
    2: flip_vertical_image,
    3: rotate_image,
    4: random_rotation,
    5: random_shear,
    6: random_crop,
    7: apply_blur,
    8: apply_exposure,
    9: apply_random_noise,
    10: cutout,
    11: mosaic,
    12: color_jitter,
    13: rotate_with_bounding_box,
    14: clahe_equalization,
    15: random_zoom,
    16: channel_shuffle,
    17: histogram_equalization
    }
    image_augmentation__name = {
    1: 'horizontally_flipped',
    2: 'vertically_flipped',
    3: '90_rotated',
    4: 'random_rotated',
    5: 'random_sheared',
    6: 'random_croped',
    7: 'blured',
    8: 'exposured',
    9: 'random_noised',
    10: 'cutout',
    11: 'mosaic',
    12: 'color_jittered',
    13: 'bounding_box_rotated',
    14: 'clahe_equalized',
    15: 'random_zoomed',
    16: 'channel_shuffled',
    17: 'histogram_equalized'
    }

    list_of_methods = [int(choice) for choice in user_choice]

    for filename in os.listdir(input_dir):
        for method in list_of_methods:
            # Use the get method to provide a default name if the key is not present
            method_name = image_augmentation__name.get(method, f'unknown_{method}')
            method_function = image_augmentation_functions.get(method)

            if method_function is not None:
                original_image = np.array(Image.open(os.path.join(input_dir, filename)))
                image = method_function(original_image)

                Image.fromarray(image).save(os.path.join(output_dir, f'{method_name}_{filename}'))
                print(f'Done for {filename}')
            else:
                print(f'Error: Method {method} not found.')

    var = 'DONE'
    
    
# get_custom_augmented_images(input_dir,output_dir,user_choice)
import zipfile
# zip_filename = 'augmented_images1.zip'
# output_dir = 'augmented_images'
def create_zip(zip_filename,output_dir):
    with zipfile.ZipFile(zip_filename, 'w') as zip_file:
        # Iterate through image files in the folder and add them to the zip file
        for filename in os.listdir(output_dir):
            # Construct the full path to the image
            image_path = os.path.join(output_dir, filename)

            # Add the image to the zip file with a relative path
            zip_file.write(image_path, filename)
            
# create_zip(zip_filename,output_dir)

import tarfile

def create_tar_gz(folder_path, output_filename):
    with tarfile.open(output_filename, "w:gz") as tar:
        # Walk through all files and directories in the given folder
        for root, _, files in os.walk(folder_path):
            for file in files:
                # Construct the full path to the file
                file_path = os.path.join(root, file)
                
                # Add the file to the tar archive
                tar.add(file_path, arcname=os.path.relpath(file_path, folder_path))
                
    
    
# get_custom_augmented_images('images/','augmented_images/',['1','2','4','6','10'])