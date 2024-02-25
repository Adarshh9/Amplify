# Amplify

## Image Augmentation Script

This Python script is designed to facilitate image augmentation for machine learning and computer vision projects. Image augmentation is a crucial step in the preprocessing pipeline, enhancing the diversity and quality of your training dataset. The script provides a collection of image transformation functions, allowing users to easily generate augmented images with various characteristics.

### Features

- **Customization:** Users can choose specific augmentation functions based on their dataset requirements. Each function is assigned a unique number for easy customization.

- **Flexibility:** The script supports a variety of augmentation techniques, including flips, rotations, blurs, exposure adjustments, and more. Users can experiment with different combinations to find the most suitable augmentation strategy.

- **Archiving Options:** The script includes functionality to create tar.gz archives of the augmented images, streamlining the process of organizing and sharing datasets.

- **Dependency:** The script relies on popular Python libraries such as Pillow, NumPy, and OpenCV, ensuring ease of use and compatibility.

## Requirements

Make sure you have the following libraries installed:

- Pillow (PIL)
- NumPy
- OpenCV (cv2)

You can install the required libraries using the following command:

```bash
pip install Pillow numpy opencv-python
```

## Usage

1. **Image Augmentation:**

    Modify the `input_dir`, `output_dir`, and `user_choice` variables in the script before running it. Replace the placeholders with the actual paths and choices for augmentation functions.

    ```python
    input_dir = 'images/'
    output_dir = 'augmented_images/'
    user_choice = ['1', '2', '4', '6', '10']  # Example: Choose functions by their corresponding numbers
    
    get_custom_augmented_images(input_dir, output_dir, user_choice)
    ```

2. **Creating Zip Archive:**

    Uncomment and modify the following code block to create a zip archive of the augmented images.

    ```python
    # zip_filename = 'augmented_images.zip'
    # create_zip(zip_filename, output_dir)
    ```

3. **Creating Tar.gz Archive:**

    Uncomment and modify the following code block to create a tar.gz archive of the augmented images.

    ```python
    # tar_filename = 'augmented_images.tar.gz'
    # create_tar_gz(output_dir, tar_filename)
    ```

## License

This project is licensed under the [GNU General Public License v3.0](LICENSE).

## Acknowledgments

- The script uses the Pillow and OpenCV libraries for image processing.
- Image augmentation functions are adapted from common techniques used in data augmentation for computer vision.

Feel free to customize this README to better fit your project. Add any additional information or acknowledgments as needed.

## LIVE SITE
Visit the live site [here](https://amplify-r86c.onrender.com) to utilize the image augmentation functionalities online.

