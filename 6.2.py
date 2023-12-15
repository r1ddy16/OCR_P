import os
import json
import cv2
import easyocr
import numpy as np
import spacy
from PIL import Image, ImageFont, ImageDraw
import matplotlib.pyplot as plt
import glob
import pytesseract

# Load the custom Finnish font (Arial)
font_size = 24  # Adjust the font size as needed
font = ImageFont.truetype("arial.ttf", font_size)

# Load the Finnish language model
nlp = spacy.load('fi_core_news_sm')

def preprocess_image(img, grayscale=False, bilateral_filter=False, clahe=False):
    # Check if the image is already in grayscale
    if len(img.shape) == 3 and img.shape[2] != 1:  # Check for 3 channels (e.g., RGB)
        if grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if bilateral_filter:
        img = cv2.bilateralFilter(img, 9, 75, 75)

    if clahe:
        # Ensure the image is in grayscale before applying CLAHE
        if len(img.shape) == 3 and img.shape[2] != 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe_obj = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(256, 256))
        img = clahe_obj.apply(img)

    return img

def annotate_and_display_image(image_path, result, output_img, resolution, resize_factor=0.5):
    img = cv2.imread(image_path)
    word_coordinates = {
         "items": []  # existing items
    }
    for detection in result:
        coords = detection[0]
        text = detection[1]
        cv2.polylines(img, [np.array(coords, np.int32).reshape((-1, 1, 2))], True, (0, 255, 0), 2)

        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        # Shift text position slightly above the top-left corner of the box
        text_position = (coords[0][0], coords[0][1] - font_size)
        if text_position[1] < 0:  # If shifting up goes out of image, shift down instead
            text_position = (coords[0][0], coords[2][1])

        draw.text(text_position, text, font=font, fill=(0, 0, 255, 255))

        img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        word_data = {
            "image": os.path.basename(image_path),
            "text": text,
            "resolution": resolution,
            "x0": int(coords[0][0]),
            "y0": int(coords[0][1]),
            "x1": int(coords[1][0]),
            "y1": int(coords[1][1]),
            "x2": int(coords[2][0]),
            "y2": int(coords[2][1]),
            "x3": int(coords[3][0]),
            "y3": int(coords[3][1]),
        }
        word_coordinates["items"].append(word_data)

    # Resize the annotated image
    height, width = img.shape[:2]
    new_width = int(width * resize_factor)
    new_height = int(height * resize_factor)
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Display the resized image
    plt.imshow(cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.draw()
    plt.pause(2)
    plt.close(plt.gcf())

    # Save the resized image
    cv2.imwrite(output_img, resized_img)

    return word_coordinates

def annotate_image_with_text(image_path, text_data, output_img):
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)

    for item in text_data:
        text = item['text']
        coords = item['coordinates']

        draw.rectangle([(coords['x0'], coords['y0']), (coords['x2'], coords['y2'])], outline="green")

        # Shift text position slightly above the top-left corner of the box
        text_position = (coords['x0'], coords['y0'] - font_size)
        if text_position[1] < 0:  # If shifting up goes out of image, shift down instead
            text_position = (coords['x0'], coords['y2'])

        try:
            draw.text(text_position, text, fill="blue", font=font)
        except UnicodeEncodeError:
            # Handle the exception if needed
            pass

    img.save(output_img)


def detect_skew_angle_hough(image):
    """
    Detect the skew angle of an image using Hough Line Transform.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Apply Hough Line Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    # Calculate angles
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        angles.append(angle)

    # Average angle of all lines
    median_angle = np.median(angles)
    return median_angle


def rotate_image(image, angle):
    """
    Rotate the image around its center.
    """
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # Rotate the entire image
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return rotated

def correct_skew(image):
    """
    Correct the skew of an image given an angle.
    """
    # Detect skew angle
    skew_angle = detect_skew_angle_hough(image)

    # Correct the skew
    corrected_image = rotate_image(image, -skew_angle)

    return corrected_image

# New Functions for Tesseract OCR with Cropping
def crop_image_sections(img_path, number_of_sections):
    image = Image.open(img_path)
    width, height = image.size

    section_width = width // number_of_sections
    sections = []
    section_offsets = []
    for i in range(number_of_sections):
        start_x = i * section_width
        end_x = start_x + section_width if (i < number_of_sections - 1) else width
        section_offsets.append(start_x)  # Save the x-offset for each section
        crop = image.crop((start_x, 0, end_x, height))
        sections.append(crop)
    return sections, section_offsets

def ocr_sections_with_coordinates(sections, section_offsets):
    text_results = []
    for section_index, section in enumerate(sections):
        data = pytesseract.image_to_data(section, lang='eng+fin', output_type=pytesseract.Output.DICT)
        offset_x = section_offsets[section_index]  # Get the x offset for the current section
        for i in range(len(data['text'])):
            if int(data['conf'][i]) > 0 and len(data['text'][i].strip()) > 1:
                x, y, w, h = data['left'][i] + offset_x, data['top'][i], data['width'][i], data['height'][i]
                text_results.append({
                    "text": data['text'][i].strip(),
                    "coordinates": {
                        "x0": x, "y0": y,
                        "x1": x + w, "y1": y,
                        "x2": x + w, "y2": y + h,
                        "x3": x, "y3": y + h
                    }
                })
    return text_results


# Main execution
if __name__ == "__main__":
    folder_path = '*:\\*\\*'
    image_paths = [f for f in glob.glob(os.path.join(folder_path, "*.jpg"))]
    output_dir = "*:\\*test\\*\\result_filter"
    output_img_prefix = os.path.join(output_dir, 'annotated_image')
    reader = easyocr.Reader(['en', 'et'])

    all_word_coordinates = {
        "cycle_0": [],
        "cycle_1": [],
        "cycle_2": [],
        "cycle_3": [],
        "with_rotation": [],
        "skew_corrected": [],
        "tesseract": []
    }

    # First cycle (no rotation)
    # Define four sets of preprocessing parameters and confidence thresholds
    preprocessing_settings = [
        {'grayscale': False, 'bilateral_filter': False, 'clahe': False},
        {'grayscale': True, 'bilateral_filter': False, 'clahe': False},
        {'grayscale': True, 'bilateral_filter': False, 'clahe': True},
        {'grayscale': True, 'bilateral_filter': True, 'clahe': True}
    ]

    confidence_thresholds = [0.1,0.3,0.3,0.3]  # Define four confidence thresholds

    # Iterate through each set of parameters
    for cycle_index, (settings, confidence_threshold) in enumerate(zip(preprocessing_settings, confidence_thresholds)):
        for index, image_path in enumerate(image_paths):
            img = cv2.imread(image_path)
            if img is None:
                print(f"Error: Unable to load the image {image_path}. Skipping to the next image.")
                continue
            resolution = f"x: {img.shape[1]} , y: {img.shape[0]} "  # (width, height)
            # Apply preprocessing with current settings
            img = preprocess_image(img, grayscale=settings['grayscale'],
                                   bilateral_filter=settings['bilateral_filter'], clahe=settings['clahe'])
            result = reader.readtext(img, paragraph=False)

            filtered_results = [detection for detection in result if detection[2] >= confidence_threshold]

            output_img = f'{output_img_prefix}_cycle_{cycle_index}_image_{index}.jpg'
            word_coordinates = annotate_and_display_image(image_path, filtered_results, output_img, resolution)
            all_word_coordinates[f"cycle_{cycle_index}"].append(word_coordinates)

    # Second cycle (with more rotation)
    for index, image_path in enumerate(image_paths):
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Unable to load the image {image_path}.")
            continue  # Skip to the next iteration if the image fails to load

        # The resolution should be retrieved here, after confirming img is not None
        resolution = f"x: {img.shape[1]} , y: {img.shape[0]} "  # (width, height)

        # Apply preprocessing here
        img = preprocess_image(img, grayscale=True, bilateral_filter=False, clahe=True)  # Include more rotation angles

        rotation_angles = [70, 80, 90, 100, 110, 160, 170, 180, 190, 200, 250, 260, 270, 280, 290]
        result = reader.readtext(img, rotation_info=rotation_angles, paragraph=False)

        confidence_threshold = 0.2
        filtered_results = [detection for detection in result if detection[2] >= confidence_threshold]

        output_img = f'{output_img_prefix}_with_more_rotation_{index}.jpg'
        word_coordinates = annotate_and_display_image(image_path, filtered_results, output_img, resolution)
        all_word_coordinates['with_rotation'].append(word_coordinates)

    # Third cycle (skew correction)
    for index, image_path in enumerate(image_paths):
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Unable to load the image {image_path}. Skipping to the next image.")
            continue  # Skip to the next iteration if the image fails to load
        resolution = f"({img.shape[1]},{img.shape[0]})"  # (width, height)
        skew_angle = detect_skew_angle_hough(img)
        print(f"Detected skew angle for {image_path}: {skew_angle}")

        corrected_img = correct_skew(img)
        # After correction, you might want to save this image temporarily to inspect the correction
        cv2.imwrite(f'{output_img_prefix}_skew_corrected_temp_{index}.jpg', corrected_img)

        # Preprocess the corrected image if necessary
        corrected_img = preprocess_image(corrected_img, grayscale=True, bilateral_filter=False, clahe=True)

        # Now process the corrected image with OCR
        result = reader.readtext(corrected_img, paragraph=False)

        confidence_threshold = 0.4
        filtered_results = [detection for detection in result if detection[2] >= confidence_threshold]
        # Debug: If no results, print a message.
        if not filtered_results:
            print(f"No results after skew correction for {image_path}")
        output_img = f'{output_img_prefix}_skew_corrected_{index}.jpg'
        word_coordinates = annotate_and_display_image(image_path, filtered_results, output_img, resolution)
        all_word_coordinates["skew_corrected"].append(word_coordinates)
    # Fourth cycle (Tesseract OCR with Cropping and Coordinates)
    for index, image_path in enumerate(image_paths):
        sections, section_offsets = crop_image_sections(image_path,18)  # Modify the crop function to also return offsets
        text_data = ocr_sections_with_coordinates(sections, section_offsets)
        # Annotate the image with recognized text
        output_img = f'{output_img_prefix}_tesseract_annotated_{index}.jpg'
        annotate_image_with_text(image_path, text_data, output_img)
        # Combine the text data for JSON
        texts = [item['text'] for item in text_data]
        coordinates = [item['coordinates'] for item in text_data]
        all_word_coordinates["tesseract"].append({
            "image": os.path.basename(image_path),
            "text": "\n".join(texts),
            "coordinates": coordinates
        })
    # Save the results to a JSON file
    with open(os.path.join(output_dir, 'results.json'), 'w', encoding='utf-8') as json_file:
        json.dump(all_word_coordinates, json_file, ensure_ascii=False, indent=4)
