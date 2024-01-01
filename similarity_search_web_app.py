import streamlit as st
import pandas as pd
from PIL import Image
import requests
import os
import shutil
from io import BytesIO
import torchvision.transforms as transforms
from torchvision.models import resnet50
from annoy import AnnoyIndex
import matplotlib.pyplot as plt

def extract_features(image_path, model, transform):
    image = Image.open(image_path)
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    image = transform(image).unsqueeze(0)
    feature_vector = model(image).detach().numpy().flatten()
    return feature_vector
# this function is for testing the model images will be rendered in the streamlit application
def show_images(image_paths):
    num_columns=3
    num_images = len(image_paths)
    num_rows = (num_images -1) // num_columns + 1
    plt.figure(figsize=(15, 7))
    for i, image_path in enumerate(image_paths):
        plt.subplot(num_rows, num_columns, i + 1)
        image = Image.open(image_path)
        plt.imshow(image)
        plt.axis('off')
        plt.title(f'Image {i+1}')
    plt.subplots_adjust(wspace=0.2, hspace=0.3)
    plt.show()

def make_dataset(csv_file_path):
    if os.path.exists("Data"):
        shutil.rmtree("Data")
    os.mkdir("Data")
    df = pd.read_csv(csv_file_path)
    num_features = 1000  
    annoy_index = AnnoyIndex(num_features)
    model = resnet50(pretrained=True)
    model.eval()
    # performing preprocessing on the images to reduce dimensions and normalization.
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    for i, row in df.iterrows():
        product_id = row['Product ID']
        image_url = row['image_link']
        image_path = os.path.join("Data", f"image_{product_id}.jpg") 
        response = requests.get(image_url)
        if response.status_code == 200:
            image_data = BytesIO(response.content)
            image = Image.open(image_data)
            if image.mode == 'RGBA':
                image = image.convert('RGB')
            image.save(image_path, "JPEG") 
            feature_vector = extract_features(image_path, model, transform)
            annoy_index.add_item(i, feature_vector)
        else:
            print(f"download failed for image with url: {image_url}")
    annoy_index.build(20) # building with 20 subtrees of features for more accuracy this can be slower at times but accuracy is prioritized in systems like this as well so using a tradeoff value of 20
    annoy_index.save('image_feature_index.ann')

    print(f"Dataset made with {len(os.listdir('Data'))} images")

# Part 3: Backend Implementation
# Implement your image similarity calculation logic here

def new_image_search(uploaded_image, num_neighbors=1):
    model = resnet50(pretrained=True)
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_image = Image.open(uploaded_image)
    if test_image.mode == 'RGBA':
        test_image = test_image.convert('RGB')
    test_image = transform(test_image).unsqueeze(0)
    test_feature_vector = model(test_image).detach().numpy().flatten()
    index = AnnoyIndex(1000) # 1000 is the number of features extracted from the resnet model
    index.load("./image_feature_index.ann")
    similar_image_indices = index.get_nns_by_vector(test_feature_vector, num_neighbors)
    similar_image_paths = [os.path.join("Data", f"image_{i+1}.jpg") for i in similar_image_indices]
    return similar_image_paths

# Part 9: Submission
# Define a function to display the assignment report
def display_assignment_report():
    st.title("Image Similarity Finder Web App")
    st.header("Assignment Report")

    # Add your report content here

# Part 4: Frontend Web App
# Create a Streamlit app
def main():        
    # making the dataset
    # make_dataset("dataset.csv") # run this only once to download the dataset and make the annoy feature vector database which is saved to the local directory for the similarity model

    st.title("Image Similarity Finder")
    st.write("Discover visually similar images in the dataset.")
    st.sidebar.header("Instructions")
    st.sidebar.markdown("1. Upload an image.")
    st.sidebar.markdown("2. Enter the number of similar images you want in the textbox")
    st.sidebar.markdown("2. Click the 'Find Similar Images' button.")

    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image:
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        num_similar_images = st.text_input("Number of similar images needed:", value="5")
        # Perform image similarity search
        if st.button("Find Similar Images", key="find_similar"):
            st.info("Searching for similar images...")

            similar_image_paths = new_image_search(uploaded_image,int(num_similar_images))
            st.header("Similar Images")
            for img_path in similar_image_paths:
                img = Image.open(img_path)
                st.image(img, caption="Similar Image", use_column_width=True)


    

    st.sidebar.header("Screenshots")
    screenshots = os.listdir("Screenshots")
    for screenshot in screenshots:
        screenshot_path = os.path.join("Screenshots", screenshot)
        st.sidebar.image(screenshot_path, caption=screenshot, use_column_width=True)


    # test code
    # test_image_path = "Data/image_130.jpg"
    # similar_image_paths = new_image_search(test_image_path)
    # print(similar_image_paths)
    # show_images(similar_image_paths)
    # st.title("Image Similarity Finder")
    # Part 11: Provide instructions to users
    # st.sidebar.header("Instructions")
    # st.sidebar.markdown("1. Upload an image.")
    # st.sidebar.markdown("2. Click the 'Find Similar Images' button.")

    # # Upload image
    # uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    # if uploaded_image:
    #     # Display the uploaded image
    #     st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    #     # Perform image similarity search
    #     if st.button("Find Similar Images"):
    #         similar_image_paths = new_image_search(uploaded_image)
    #         st.header("Similar Images")
    #         for img_path in similar_image_paths:
    #             img = Image.open(img_path)
    #             st.image(img, caption="Similar Image", use_column_width=True)


    # # Part 10: Explain the structure and layout of your Streamlit app
    # The Streamlit app features a clear title and description, concise sidebar instructions, and an image upload section. Users can trigger the similarity search with a prominent button, and search feedback is provided. The users can also choose the number of similar images they want by entering the value in the textbox. Similar images are displayed with clear headers for easy exploration.

    # Part 18: User Interface Design
    # The user interface design in this application emphasizes simplicity and accessibility. It provides a clear title and description, user-friendly sidebar instructions, and a straightforward upload feature for image selection. The textbox to enter the number of similar images required allows the user to input how many similar images they want to see. The "Find Similar Images" button initiates the similarity search process and provides informative feedback. Similar images are presented with clear headers and are easily viewable. This design ensures an intuitive and efficient user experience for discovering visually related images from the dataset.
    

if __name__ == "__main__":
    main()
