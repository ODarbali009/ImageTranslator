import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from collections import Counter
from sklearn.cluster import KMeans
import numpy as np
import cv2
import easyocr
from googletrans import Translator
import os
import asyncio
import traceback


src_lang = 'ja'
dst_lang = 'en'
threshold = 0.002

def get_background_and_text_colors(image):

    image = image.resize((100, 100))

    pixels = list(image.getdata())

    color_counts = Counter(pixels)

    most_common_colors = color_counts.most_common(1)

    background_color = most_common_colors[0][0]
    text_color = tuple(map(int, get_colors(image, background_color, 3)))

    return background_color, text_color

def get_colors(image, background_color, number_of_colors):

    modified_image = cv2.resize(np.array(image), (600, 400), interpolation = cv2.INTER_AREA)

    modified_image = modified_image.reshape(modified_image.shape[0]*modified_image.shape[1], 3)

    clf = KMeans(n_clusters = number_of_colors)
    labels = clf.fit_predict(modified_image)

    counts = Counter(labels)

    center_colors = clf.cluster_centers_
    ordered_colors = [center_colors[i]/255 for i in counts.keys()]
    rgb_colors = [ordered_colors[i]*255 for i in counts.keys()]


    rgb_colors = np.array(rgb_colors)

    first_color = np.array(background_color)


    distances = np.linalg.norm(rgb_colors[:] - first_color, axis=1)

    max_distance_index = np.argmax(distances)

    most_distant_color = rgb_colors[max_distance_index]

    return most_distant_color

def create_text_image(size, background_color, text_color, text):
    width, height = size
    new_image = Image.new("RGB", size, background_color)
    draw = ImageDraw.Draw(new_image)

    max_text_width = width * 0.99
    max_text_height = height * 0.99

    font_size = 100
    while font_size > 1:
        font = ImageFont.truetype("arial.ttf", font_size, layout_engine=ImageFont.Layout.BASIC)
        
        text_width, text_height = draw.textbbox((0, 0), text, font=font)[2:4]
        
        if text_width <= max_text_width and text_height <= max_text_height:
            break
        else:
            font_size -= 1

    font = ImageFont.truetype("arial.ttf", font_size, layout_engine=ImageFont.Layout.BASIC)
    
    text_width, text_height = draw.textbbox((0, 0), text, font=font)[2:4]

    y_offset = (height - text_height) // 2

    x_offset = (width - text_width) // 2

    draw.text((x_offset, y_offset), text, fill=text_color, font=font)

    return new_image


def enhance_image_quality(image_path):
    pil_image = Image.open(image_path).convert('RGB')

    width, height = pil_image.size
    pil_image = pil_image.resize((int(width * 2), int(height * 2)), Image.LANCZOS)

    pil_image = pil_image.filter(ImageFilter.SHARPEN)

    return pil_image

async def detect_and_translate_text(image_path):
    try:
        reader = easyocr.Reader([src_lang])
        pil_image = enhance_image_quality(image_path)
        image_array = np.array(pil_image)
        results = reader.readtext(image_array)
        translator = Translator()
        processed_images = []

        for (bbox, text, prob) in results:
            if text.strip() and prob > threshold:
                print(f"Detected text: {text}, Confidence: {prob}")
                translated_text = await translator.translate(text, src=src_lang, dest=dst_lang)
                translated_text = translated_text.text
                x_min, y_min = map(int, bbox[0])
                x_max, y_max = map(int, bbox[2])
                
                MIN_CROP_SIZE = 5
                if x_max - x_min < MIN_CROP_SIZE or y_max - y_min < MIN_CROP_SIZE:
                    print(f"Skipping bounding box smaller than {MIN_CROP_SIZE}px.")
                    continue

                i_s = pil_image.crop((x_min, y_min, x_max, y_max))
                background_color, text_color = get_background_and_text_colors(i_s)
                o_f = create_text_image(i_s.size, background_color, text_color, translated_text)
                processed_images.append((o_f, (x_min, y_min, x_max, y_max)))

        for processed_image, box in processed_images:
            pil_image.paste(processed_image, box)

        output_folder = "output"
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, f"translated_{os.path.basename(image_path)}")
        pil_image.save(output_path)
        return pil_image, output_path

    except Exception as e:
        error_msg = traceback.format_exc()
        print(f"Error: {error_msg}")
        raise RuntimeError(f"An error occurred during text detection: {error_msg}")



# Streamlit app
def main():
    st.title("Text Detection and Translation")
    st.write("Upload an image with Japanese text, and we will translate it to English.")

    uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        input_folder = "input"
        os.makedirs(input_folder, exist_ok=True)
        input_path = os.path.join(input_folder, uploaded_file.name)

        with open(input_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)


        if st.button("Translate Text"):
            with st.spinner("Processing..."):
                translated_image, output_path = asyncio.run(detect_and_translate_text(input_path))

                # st.image(translated_image, caption="Translated Image", use_column_width=True)
                st.image(translated_image, caption="Translated Image", use_container_width=True)
                st.write(f"Translation saved to: {output_path}")

                with open(output_path, "rb") as file:
                    btn = st.download_button(
                        label="Download Translated Image",
                        data=file,
                        file_name=f"translated_{uploaded_file.name}",
                        mime="image/jpeg"
                    )

if __name__ == "__main__":
    main()
