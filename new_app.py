import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import streamlit as st
from PIL import Image
import pandas as pd
import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader

import folium
from streamlit_folium import folium_static
from torchvision import transforms
from torch.utils.data import random_split
import time
from datetime import datetime
from fpdf import FPDF
import base64
import zipfile

# Custom Imports
from dataset import SatelliteImageDataset
from unet import UNet
from res_unet_a import ResUNetA
from utils import one_hot_to_image

# Constants
COLOR_MAP = {
    (60, 16, 152): 0,  # Building
    (132, 41, 246): 1,  # Land
    (110, 193, 228): 2,  # Road
    (254, 221, 58): 3,  # Vegetation
    (226, 169, 41): 4,  # Water
    (155, 155, 155): 5  # Other
}

MASK_LABELS = ['Building', 'Land', 'Road', 'Vegetation', 'Water', 'Other']
COLORS = ['#3C1098', '#8429F6', '#6EC1E4', '#FEDD3A', '#E2A929', '#9B9B9B']
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Paths
IMAGE_DIR = "/Users/yashas.m/Documents/Satellite-Based Geographical Poverty Predictions/final_more/images"
MASK_DIR = "/Users/yashas.m/Documents/Satellite-Based Geographical Poverty Predictions/final_more/masked"
UNET_CHECKPOINT = "/Users/yashas.m/Documents/Satellite-Based Geographical Poverty Predictions/unet/dk_unet_100ep_checkpoint.pth"
RESUNET_CHECKPOINT = "/Users/yashas.m/Documents/Satellite-Based Geographical Poverty Predictions/res_unet_a/res_unet_a_100ep_checkpoint (1).pth"

# Ensure paths exist
assert os.path.exists(UNET_CHECKPOINT), "❌ UNet checkpoint not found!"
assert os.path.exists(RESUNET_CHECKPOINT), "❌ ResUNetA checkpoint not found!"

# Cache model loading
@st.cache_resource
def load_model(model_name):
    if model_name == "UNet":
        checkpoint_path = UNET_CHECKPOINT
        model = UNet(3, 6).to(DEVICE)
    else:
        checkpoint_path = RESUNET_CHECKPOINT
        model = ResUNetA(3, 6).to(DEVICE)

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()  # Set model to evaluation mode
    return model

# Utility Functions
def preprocess_image(image):
    """Preprocess image for model input."""
    image = image.convert("RGB")  # Ensure RGB format
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])
    tensor = transform(image).unsqueeze(0).to(DEVICE)  # Add batch dimension
    return tensor

def predict_image(image_tensor, model):
    """Predict segmentation mask."""
    with torch.no_grad():
        y_pred = model(image_tensor)
        
        pred_img = one_hot_to_image(y_pred, COLOR_MAP, DEVICE)
        return pred_img.squeeze(0).cpu().permute(1, 2, 0).numpy()

def extract_class_masks(pred_img_np):
    """Extract binary masks for each class."""
    return {label: (np.all(pred_img_np == np.array(color), axis=-1).astype(np.uint8) * 255)
            for color, label in COLOR_MAP.items()}

def calculate_white_percentage(masks):
    """Calculate the percentage of white pixels for each class."""
    return {label: (np.sum(mask == 255) / mask.size) * 100 for label, mask in masks.items()}

def classify_poverty(white_percentages):
    """Determine poverty level based on feature percentages."""
    # Calculate based on feature distribution without confidence
    
    if white_percentages[1] > 70:  # Land
        return "🔴 Below Geographical Poverty", "red"
    
    total_features = sum(white_percentages[label] for label in [0, 2, 3, 4])
    
    if total_features > 40:
        return "🟢 Above Geographical Poverty", "green"
    elif 20 <= total_features <= 40:
        return "🟠 Average Geographical Poverty", "orange"
    
    return "🔴 Below Geographical Poverty", "red"

def display_class_masks(masks):
    """Show class-wise masks."""
    st.subheader("📌 Class-wise Masks")
    fig, axes = plt.subplots(1, len(masks), figsize=(15, 5))
    for idx, (label, mask) in enumerate(masks.items()):
        ax = axes[idx]
        ax.imshow(mask, cmap="gray")
        ax.set_title(MASK_LABELS[label])
        ax.axis("off")
    st.pyplot(fig)
    return fig

def display_pie_chart(white_percentages):
    """Show feature distribution as a pie chart."""
    st.subheader("📊 Feature Distribution")
    sizes = [white_percentages.get(label, 0) for label in range(len(MASK_LABELS))]
    fig, ax = plt.subplots(figsize=(8, 6))
    wedges, _ = ax.pie(sizes, colors=COLORS, startangle=140, wedgeprops={"edgecolor": "black"})
    legend_labels = [f"{MASK_LABELS[i]}: {size:.2f}%" for i, size in enumerate(sizes)]
    ax.legend(wedges, legend_labels, title="Features", loc="center left", bbox_to_anchor=(1, 0.5))
    ax.axis("equal")
    st.pyplot(fig)
    return fig

def create_pdf_report(image, pred_img, class_masks_fig, pie_chart_fig, white_percentages, poverty_status):
    """Create a downloadable PDF report."""
    pdf = FPDF()
    pdf.add_page()
    
    # Title
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(200, 10, 'Satellite-Based Poverty Prediction Report', 0, 1, 'C')
    pdf.ln(10)
    
    # Date and time
    pdf.set_font('Arial', '', 10)
    pdf.cell(200, 10, f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1, 'R')
    pdf.ln(5)
    
    # Save figures to temporary files
    temp_image = "temp_image.png"
    temp_pred = "temp_pred.png"
    temp_masks = "temp_masks.png"
    temp_pie = "temp_pie.png"
    
    # Save input image and prediction
    plt.figure(figsize=(5, 5))
    plt.imshow(np.array(image))
    plt.axis('off')
    plt.title("Input Satellite Image")
    plt.tight_layout()
    plt.savefig(temp_image)
    
    plt.figure(figsize=(5, 5))
    plt.imshow(pred_img)
    plt.axis('off')
    plt.title("Segmentation Result")
    plt.tight_layout()
    plt.savefig(temp_pred)
    
    # Save other figures (already created)
    class_masks_fig.savefig(temp_masks)
    pie_chart_fig.savefig(temp_pie)
    
    # Add images to PDF
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(200, 10, 'Images', 0, 1, 'L')
    pdf.image(temp_image, x=10, y=pdf.get_y(), w=90)
    pdf.image(temp_pred, x=110, y=pdf.get_y(), w=90)
    pdf.ln(100)  # Space for images
    
    # Add class masks
    pdf.add_page()
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(200, 10, 'Class-wise Masks', 0, 1, 'L')
    pdf.image(temp_masks, x=10, y=pdf.get_y(), w=180)
    pdf.ln(70)  # Space for image
    
    # Add pie chart
    pdf.cell(200, 10, 'Feature Distribution', 0, 1, 'L')
    pdf.image(temp_pie, x=30, y=pdf.get_y(), w=150)
    pdf.ln(80)  # Space for image
    
    # Add detailed statistics
    pdf.add_page()
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(200, 10, 'Detailed Analysis Results', 0, 1, 'L')
    pdf.ln(5)
    
    # Feature percentages
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(200, 10, 'Feature Percentages:', 0, 1, 'L')
    pdf.set_font('Arial', '', 10)
    
    for i, label in enumerate(MASK_LABELS):
        percentage = white_percentages.get(i, 0)
        pdf.cell(200, 8, f'{label}: {percentage:.2f}%', 0, 1, 'L')
    
    pdf.ln(5)
    
    # Poverty classification
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(200, 10, 'Poverty Classification:', 0, 1, 'L')
    pdf.set_font('Arial', '', 12)
    pdf.cell(200, 10, f'{poverty_status}', 0, 1, 'L')
    
    # Clean up temporary files
    for temp_file in [temp_image, temp_pred, temp_masks, temp_pie]:
        if os.path.exists(temp_file):
            os.remove(temp_file)
    
    return pdf.output(dest='S').encode('latin1')

def batch_process_images(uploaded_files, model):
    """Process multiple images and return results."""
    batch_results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, uploaded_file in enumerate(uploaded_files):
        status_text.text(f"Processing image {i+1}/{len(uploaded_files)}")
        image = Image.open(uploaded_file)
        
        # Preprocess and predict
        image_tensor = preprocess_image(image)
        pred_img_np = predict_image(image_tensor, model)
        
        # Extract class masks and percentages
        class_masks = extract_class_masks(pred_img_np)
        white_percentages = calculate_white_percentage(class_masks)
        
        # Classification
        poverty_status, color = classify_poverty(white_percentages)
        
        batch_results.append({
            "filename": uploaded_file.name,
            "image": image,
            "prediction": pred_img_np,
            "white_percentages": white_percentages,
            "poverty_status": poverty_status,
        })
        
        progress_bar.progress((i + 1) / len(uploaded_files))
    
    status_text.text("Processing complete!")
    return batch_results

# Application UI
st.set_page_config(page_title="Satellite Poverty Analysis", layout="wide")

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 36px;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 30px;
        text-shadow: 1px 1px 2px #ccc;
    }
    .subheader {
        font-size: 24px;
        font-weight: bold;
        color: #333;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .info-box {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
        border: 1px solid #e0e0e0;
    }
    .metrics-container {
        background-color: #f0f7ff;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
    }
    .footer {
        text-align: center;
        margin-top: 50px;
        padding: 20px;
        border-top: 1px solid #eee;
        font-size: 12px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🌍 Satellite-Based Geographical Poverty Predictions</p>', unsafe_allow_html=True)

# Create tabs for different functionalities
tab1, tab2, tab3, tab4 = st.tabs(["📊 Single Image Analysis", "📁 Batch Processing", "🧠 Model Information", "📍 Location Analysis"])

# Tab 1: Single Image Analysis
# Function to create PDF report
def create_pdf_report(image, prediction, mask_fig, pie_fig, percentages, poverty_status):
    # Create a byte stream to save the PDF to
    buffer = io.BytesIO()
    
    # Create a PDF document
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    
    # Add title
    c.setFont("Helvetica-Bold", 20)
    c.drawString(50, height - 50, "Satellite Image Poverty Analysis Report")
    
    # Add date
    c.setFont("Helvetica", 12)
    c.drawString(50, height - 70, f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Add original image
    img_bytes = io.BytesIO()
    img_pil = Image.fromarray(np.array(image))
    img_pil = img_pil.resize((300, 300))
    img_pil.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    c.drawImage(ImageReader(img_bytes), 50, height - 380, width=300, height=300)
    c.drawString(50, height - 390, "Original Satellite Image")
    
    # Add prediction image
    pred_bytes = io.BytesIO()
    pred_pil = Image.fromarray(prediction.astype('uint8'))
    pred_pil = pred_pil.resize((300, 300))
    pred_pil.save(pred_bytes, format='PNG')
    pred_bytes.seek(0)
    c.drawImage(ImageReader(pred_bytes), width-350, height - 380, width=300, height=300)
    c.drawString(width-350, height - 390, "Segmentation Prediction")
    
    # Add class percentages
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 420, "Feature Distribution:")
    c.setFont("Helvetica", 12)
    y_pos = height - 440
    for class_name, percentage in percentages.items():
        c.drawString(70, y_pos, f"{class_name}: {percentage:.2f}%")
        y_pos -= 20
    
    # Add poverty classification
    c.setFont("Helvetica-Bold", 14)
    c.drawString(width-350, height - 420, "Poverty Classification:")
    
    # Set color based on classification
    if "Low" in poverty_status:
        c.setFillColorRGB(0, 0.5, 0)  # Dark green
    elif "Medium" in poverty_status:
        c.setFillColorRGB(1, 0.65, 0)  # Orange
    else:
        c.setFillColorRGB(0.8, 0, 0)  # Dark red
    
    c.setFont("Helvetica-Bold", 12)
    c.drawString(width-330, height - 440, poverty_status)
    c.setFillColorRGB(0, 0, 0)  # Reset to black
    
    # Add class masks figure
    mask_bytes = io.BytesIO()
    mask_fig.savefig(mask_bytes, format='PNG', bbox_inches='tight')
    mask_bytes.seek(0)
    c.showPage()  # New page
    c.drawImage(ImageReader(mask_bytes), 50, height - 400, width=width-100, height=350)
    c.drawString(50, height - 410, "Detailed Class Masks")
    
    # Add pie chart
    pie_bytes = io.BytesIO()
    pie_fig.savefig(pie_bytes, format='PNG', bbox_inches='tight')
    pie_bytes.seek(0)
    c.drawImage(ImageReader(pie_bytes), 50, height - 750, width=width-100, height=300)
    c.drawString(50, height - 760, "Feature Distribution Pie Chart")
    
    # Save the PDF
    c.save()
    buffer.seek(0)
    return buffer.getvalue()

# Main app content
with tab1:
    with st.sidebar:
        st.subheader("Model Settings")
        model_name = st.selectbox("Select Model", ["UNet", "ResUNet-A"])
        
        st.subheader("Analysis Options")
        show_detailed_masks = st.checkbox("Show Detailed Class Masks", True)
        
        uploaded_file = st.file_uploader("📤 Upload a Satellite Image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        try:
            # Start timing
            start_time = time.time()
            
            # Load selected model
            model = load_model(model_name)
            
            # Load uploaded image
            image = Image.open(uploaded_file)
            
            # Create columns for images
            img_col1, img_col2 = st.columns(2)
            
            img_col1.image(image, caption="🖼️ Uploaded Satellite Image", use_column_width=True)
            
            # Preprocess and predict
            image_tensor = preprocess_image(image)
            pred_img_np = predict_image(image_tensor, model)
            
            img_col2.image(pred_img_np, caption="📌 Predicted Segmentation", use_column_width=True)
            
            # Extract class masks
            class_masks = extract_class_masks(pred_img_np)
            
            # Calculate white pixel percentages
            white_percentages = calculate_white_percentage(class_masks)
            
            # Display detailed masks if option is selected
            mask_fig = None
            if show_detailed_masks:
                mask_fig = display_class_masks(class_masks)
            
            # Show pie chart
            pie_fig = display_pie_chart(white_percentages)
            
            # Poverty classification
            poverty_status, color = classify_poverty(white_percentages)
            
            # Display results in a metrics section
            st.markdown('<div class="metrics-container">', unsafe_allow_html=True)
            
            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
            
            with metrics_col1:
                st.subheader("🏠 Poverty Classification")
                st.markdown(f"<p style='color:{color}; font-size: 24px; font-weight: bold;'>{poverty_status}</p>", unsafe_allow_html=True)
            
            with metrics_col3:
                processing_time = time.time() - start_time
                st.subheader("⏱️ Processing Time")
                st.markdown(f"<p style='font-size: 24px; font-weight: bold;'>{processing_time:.2f} seconds</p>", unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Add explanations
            with st.expander("📋 Analysis Explanation"):
                st.markdown("""
                **How is poverty classified?**
                
                The model analyzes the presence and distribution of various geographical features:
                
                - **Buildings**: Dense building areas often indicate urban development
                - **Roads**: Well-connected road networks suggest better infrastructure
                - **Vegetation**: Planned green spaces often correlate with developed areas
                - **Water**: Access to water resources can indicate development
                - **Land**: Large expanses of undeveloped land may indicate lower development
                
                The algorithm weighs these features to determine the likely socioeconomic status of the region.
                """)
            
            # Generate and offer PDF report download
            if st.button("🔄 Generate Detailed PDF Report"):
                with st.spinner("Generating PDF report..."):
                    # Ensure we have all required figures
                    if mask_fig is None:
                        mask_fig = display_class_masks(class_masks)
                        st.experimental_rerun()  # Rerun to hide the automatically displayed figure
                    
                    pdf_data = create_pdf_report(
                        image, 
                        pred_img_np, 
                        mask_fig, 
                        pie_fig, 
                        white_percentages, 
                        poverty_status, 
                    )
                    
                    # Add download button for the generated PDF
                    st.download_button(
                        label="📥 Download PDF Report",
                        data=pdf_data,
                        file_name="satellite_analysis_report.pdf",
                        mime="application/pdf"
                    )
                    
        except Exception as e:
            st.error(f"⚠️ Error: {str(e)}")
            st.text("Check console/logs for details.")
    else:
        # Display sample images or instructions when no image is uploaded
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("""
        ### 📸 Upload a satellite image to begin analysis
        
        This tool analyzes satellite imagery to predict poverty levels based on geographical features:
        
        - 🏠 Buildings and infrastructure
        - 🌳 Vegetation and natural resources
        - 🛣️ Roads and transportation networks
        - 💧 Water bodies and access
        - 🏞️ Land use patterns
        
        The AI model identifies these features and uses their distribution to estimate poverty levels.
        """)
        st.markdown('</div>', unsafe_allow_html=True)


# Tab 2: Batch Processing
with tab2:
    st.markdown('<p class="subheader">📁 Batch Process Multiple Images</p>', unsafe_allow_html=True)
    
    batch_model_name = st.selectbox("Select Model for Batch Processing", ["UNet", "ResUNet-A"], key="batch_model")
    batch_files = st.file_uploader("Upload Multiple Images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    
    if batch_files and len(batch_files) > 0:
        # Show the number of uploaded images
        st.write(f"📊 {len(batch_files)} images uploaded for processing")
        
        if st.button("🔄 Process All Images"):
            # Load model for batch processing
            batch_model = load_model(batch_model_name)
            
            # Process all images
            with st.spinner("Processing batch of images..."):
                batch_results = batch_process_images(batch_files, batch_model)
            
            # Display summary table
            st.markdown('<p class="subheader">📊 Batch Processing Results</p>', unsafe_allow_html=True)
            
            # Create summary DataFrame
            summary_data = []
            for result in batch_results:
                summary_data.append({
                    "Filename": result["filename"],
                    "Poverty Status": result["poverty_status"],
                    "Buildings (%)": f"{result['white_percentages'].get(0, 0):.2f}%",
                    "Land (%)": f"{result['white_percentages'].get(1, 0):.2f}%",
                    "Roads (%)": f"{result['white_percentages'].get(2, 0):.2f}%",
                    "Vegetation (%)": f"{result['white_percentages'].get(3, 0):.2f}%",
                    "Water (%)": f"{result['white_percentages'].get(4, 0):.2f}%"
                })
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df)
    else:
        st.info("Upload multiple satellite images to process them in batch and generate comparative results.")
# Tab 3: Model Information 
with tab3:
    st.markdown('<p class="subheader">🧠 Model Information & Performance</p>', unsafe_allow_html=True)
    
    # Create tabs for different model information
    model_tab1, model_tab2, model_tab3 = st.tabs(["📊 Performance Metrics", "🔍 Architecture", "📝 Training Details"])
    
    with model_tab1:
        st.markdown("### Model Accuracy Comparison")
        
        # Sample data for demonstration (replace with actual performance metrics)
        model_metrics = pd.DataFrame({
            'Model': ['UNet', 'ResUNet-A'],
            'Accuracy': [0.81, 0.86],
            'Processing Time (ms)': [145, 170]  # Added this column for the second chart
        })
        
        st.dataframe(model_metrics)
        
        # Visualization of metrics
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        
        # Accuracy metrics - Fix for the error
        metrics = ['Accuracy']
        x = np.arange(len(metrics))
        bar_width = 0.25
        
        # Plotting only the Accuracy column
        ax[0].bar(x - bar_width/2, [model_metrics.iloc[0, 1]], bar_width, label='UNet')
        ax[0].bar(x + bar_width/2, [model_metrics.iloc[1, 1]], bar_width, label='ResUNet-A')
        ax[0].set_xticks(x)
        ax[0].set_xticklabels(metrics)
        ax[0].set_title('Accuracy Comparison')
        ax[0].set_ylim(0, 1.0)
        ax[0].legend()
        
        # Processing time
        ax[1].bar(['UNet', 'ResUNet-A'], model_metrics['Processing Time (ms)'])
        ax[1].set_ylabel('Time (ms)')
        ax[1].set_title('Processing Time Comparison')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    with model_tab2:
        st.markdown("### Model Architecture")
        st.write("The ResUNet-A architecture incorporates residual connections and attention mechanisms to improve feature extraction from satellite imagery.")
        
        # You could add architecture details or diagrams here
        st.write("""
        **UNet Architecture:**
        - Standard encoder-decoder structure
        - Skip connections between encoding and decoding paths
        - Effective for segmentation tasks
        
        **ResUNet-A Architecture:**
        - Enhanced with residual connections
        - Attention gates for feature refinement
        - Better performance on complex satellite imagery
        """)
    
    with model_tab3:
        st.markdown("### Training Details")
        st.write("Models were trained on high-resolution satellite imagery.")

# Tab 4: Location Comparison
with tab4:
    st.markdown('<p class="subheader">📍 Compare Location Over Time</p>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        This tab allows you to compare satellite imagery of the same location from different years to track changes in geographical features and poverty indicators over time.
    </div>
    """, unsafe_allow_html=True)
    
    # Location input with only Upload Images option
    st.markdown("### Upload satellite images of the same location from different years")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Earlier Year Image")
        earlier_file = st.file_uploader("Upload earlier satellite image", type=["jpg", "jpeg", "png"], key="earlier")
        earlier_year = st.number_input("Year", min_value=1980, max_value=datetime.now().year-1, value=2015, key="earlier_year")
    
    with col2:
        st.markdown("#### Later Year Image")
        later_file = st.file_uploader("Upload later satellite image", type=["jpg", "jpeg", "png"], key="later") 
        later_year = st.number_input("Year", min_value=1980, max_value=datetime.now().year, value=2023, key="later_year")
    
    model_option = st.selectbox("Select segmentation model", ["UNet", "ResUNetA"], key="compare_model")
    
    if earlier_file and later_file and st.button("Compare Images", key="compare_btn"):
        # Load model
        model = load_model(model_option)
        
        # Process earlier image
        earlier_image = Image.open(earlier_file)
        earlier_tensor = preprocess_image(earlier_image)
        earlier_pred = predict_image(earlier_tensor, model)
        earlier_masks = extract_class_masks(earlier_pred)
        earlier_percentages = calculate_white_percentage(earlier_masks)
        earlier_status, earlier_color = classify_poverty(earlier_percentages)
        
        # Process later image
        later_image = Image.open(later_file)
        later_tensor = preprocess_image(later_image)
        later_pred = predict_image(later_tensor, model)
        later_masks = extract_class_masks(later_pred)
        later_percentages = calculate_white_percentage(later_masks)
        later_status, later_color = classify_poverty(later_percentages)
        
        # Display results side by side
        st.markdown("### Comparison Results")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"#### {earlier_year}")
            st.image(earlier_image, caption="Original Image", use_column_width=True)
            st.image(earlier_pred, caption="Segmentation Result", use_column_width=True)
            st.markdown(f"**Classification: {earlier_status}**")
        
        with col2:
            st.markdown(f"#### {later_year}")
            st.image(later_image, caption="Original Image", use_column_width=True)
            st.image(later_pred, caption="Segmentation Result", use_column_width=True)
            st.markdown(f"**Classification: {later_status}**")
        
        # Feature comparison
        st.markdown("### Feature Changes Over Time")
        
        # Create feature comparison chart
        feature_comparison = pd.DataFrame({
            'Feature': MASK_LABELS,
            str(earlier_year): [earlier_percentages.get(i, 0) for i in range(len(MASK_LABELS))],
            str(later_year): [later_percentages.get(i, 0) for i in range(len(MASK_LABELS))]
        })
        
        # Calculate change
        feature_comparison['Change'] = feature_comparison[str(later_year)] - feature_comparison[str(earlier_year)]
        
        # Display table
        st.dataframe(feature_comparison.style.background_gradient(subset=['Change'], cmap='RdYlGn'))
        
        # Plot bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(MASK_LABELS))
        width = 0.35
        
        earlier_bars = ax.bar(x - width/2, feature_comparison[str(earlier_year)], width, label=str(earlier_year), color='skyblue')
        later_bars = ax.bar(x + width/2, feature_comparison[str(later_year)], width, label=str(later_year), color='coral')
        
        ax.set_ylabel('Percentage (%)')
        ax.set_title('Feature Distribution Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(MASK_LABELS)
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        st.pyplot(fig)
        
        # Summary of changes
        st.markdown("### Summary of Changes")
        
        # Calculate major changes
        significant_changes = []
        for i, label in enumerate(MASK_LABELS):
            change = feature_comparison['Change'][i]
            if abs(change) >= 5:  # Only report significant changes (>= 5%)
                direction = "increased" if change > 0 else "decreased"
                significant_changes.append(f"- {label}: {abs(change):.2f}% {direction}")
        
        if significant_changes:
            st.markdown("\n".join(significant_changes))
        else:
            st.markdown("No significant changes (≥5%) in land features detected.")
        
        # Poverty status change
        st.markdown("### Poverty Status Change")
        
        if earlier_status != later_status:
            st.markdown(f"The area has changed from **{earlier_status}** in {earlier_year} to **{later_status}** in {later_year}.")
            
            # Determine if it's an improvement or decline
            if "Above" in later_status and "Above" not in earlier_status:
                st.markdown("✅ This indicates an **improvement** in geographical poverty indicators.")
            elif "Below" in later_status and "Below" not in earlier_status:
                st.markdown("❌ This indicates a **decline** in geographical poverty indicators.")
            else:
                st.markdown("⚠️ The change in poverty classification is less definitive.")
        else:
            st.markdown(f"The poverty classification remained **{earlier_status}** from {earlier_year} to {later_year}.")
        
        # Generate comparison report
        st.markdown("### Download Comparison Report")
        
        # Create PDF comparison report
        try:
            import io
            pdf = FPDF()
            pdf.add_page()
            
            # Title
            pdf.set_font('Arial', 'B', 16)
            pdf.cell(200, 10, f' Comparison Report ({earlier_year} vs {later_year})', 0, 1, 'C')
            pdf.ln(10)
            
            # Date and time
            pdf.set_font('Arial', '', 10)
            pdf.cell(200, 10, f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1, 'R')
            pdf.ln(5)
            
            # Use BytesIO for temporary image storage
            import tempfile
            
            # Create temp directory for saving files
            with tempfile.TemporaryDirectory() as tmpdirname:
                # Save images to temporary files
                temp_earlier = os.path.join(tmpdirname, "temp_earlier.png")
                temp_later = os.path.join(tmpdirname, "temp_later.png")
                temp_earlier_pred = os.path.join(tmpdirname, "temp_earlier_pred.png")
                temp_later_pred = os.path.join(tmpdirname, "temp_later_pred.png")
                temp_chart = os.path.join(tmpdirname, "temp_chart.png")
                
                # Save input images and predictions
                plt.figure(figsize=(5, 5))
                plt.imshow(np.array(earlier_image))
                plt.axis('off')
                plt.title(f"Image from {earlier_year}")
                plt.tight_layout()
                plt.savefig(temp_earlier)
                plt.close()
                
                plt.figure(figsize=(5, 5))
                plt.imshow(earlier_pred)
                plt.axis('off')
                plt.title(f"Segmentation {earlier_year}")
                plt.tight_layout()
                plt.savefig(temp_earlier_pred)
                plt.close()
                
                plt.figure(figsize=(5, 5))
                plt.imshow(np.array(later_image))
                plt.axis('off')
                plt.title(f"Image from {later_year}")
                plt.tight_layout()
                plt.savefig(temp_later)
                plt.close()
                
                plt.figure(figsize=(5, 5))
                plt.imshow(later_pred)
                plt.axis('off')
                plt.title(f"Segmentation {later_year}")
                plt.tight_layout()
                plt.savefig(temp_later_pred)
                plt.close()
                
                # Save comparison chart
                fig.savefig(temp_chart)
                plt.close(fig)
                
                # PDF content
                pdf.add_page()
                pdf.set_font('Arial', 'B', 14)
                pdf.cell(200, 10, 'Image Comparison', 0, 1, 'L')
                
                # Add images to PDF (2x2 grid)
                pdf.image(temp_earlier, x=10, y=30, w=85)
                pdf.image(temp_earlier_pred, x=110, y=30, w=85)
                pdf.image(temp_later, x=10, y=140, w=85)
                pdf.image(temp_later_pred, x=110, y=140, w=85)
                
                # Labels
                pdf.set_font('Arial', '', 10)
                pdf.text(10, 25, f"Original Image ({earlier_year})")
                pdf.text(110, 25, f"Segmentation ({earlier_year})")
                pdf.text(10, 135, f"Original Image ({later_year})")
                pdf.text(110, 135, f"Segmentation ({later_year})")
                
                # Add classification results
                pdf.add_page()
                pdf.set_font('Arial', 'B', 14)
                pdf.cell(200, 10, 'Poverty Classification', 0, 1, 'L')
                pdf.ln(5)
                
                # Remove emoji from status text
                earlier_status_text = earlier_status.split(" ", 1)[1] if " " in earlier_status else earlier_status
                later_status_text = later_status.split(" ", 1)[1] if " " in later_status else later_status
                
                pdf.set_font('Arial', '', 12)
                pdf.cell(200, 10, f"{earlier_year}: {earlier_status_text}", 0, 1, 'L')
                pdf.cell(200, 10, f"{later_year}: {later_status_text}", 0, 1, 'L')
                pdf.ln(10)
                
                # Add comparison chart
                pdf.cell(200, 10, 'Feature Comparison Chart:', 0, 1, 'L')
                pdf.image(temp_chart, x=10, y=pdf.get_y(), w=180)
                pdf.ln(120)  # Space for chart
                
                # Feature changes table
                pdf.set_font('Arial', 'B', 12)
                pdf.cell(200, 10, 'Feature Changes:', 0, 1, 'L')
                
                # Table header
                pdf.set_font('Arial', 'B', 10)
                pdf.cell(60, 7, 'Feature', 1, 0, 'C')
                pdf.cell(40, 7, str(earlier_year), 1, 0, 'C')
                pdf.cell(40, 7, str(later_year), 1, 0, 'C')
                pdf.cell(40, 7, 'Change (%)', 1, 1, 'C')
                
                # Table data
                pdf.set_font('Arial', '', 10)
                for i, row in feature_comparison.iterrows():
                    pdf.cell(60, 7, row['Feature'], 1, 0, 'L')
                    pdf.cell(40, 7, f"{row[str(earlier_year)]:.2f}%", 1, 0, 'R')
                    pdf.cell(40, 7, f"{row[str(later_year)]:.2f}%", 1, 0, 'R')
                    
                    change = row['Change']
                    change_text = f"{change:.2f}%"
                    
                    if change > 0:
                        pdf.set_text_color(0, 128, 0)
                    elif change < 0:
                        pdf.set_text_color(255, 0, 0)
                    else:
                        pdf.set_text_color(0, 0, 0)
                        
                    pdf.cell(40, 7, change_text, 1, 1, 'R')
                    pdf.set_text_color(0, 0, 0)  # Reset color
                
                # Add a summary section
                pdf.add_page()
                pdf.set_font('Arial', 'B', 14)
                pdf.cell(200, 10, 'Summary of Changes', 0, 1, 'L')
                pdf.ln(5)
                
                # Add significant changes
                pdf.set_font('Arial', '', 12)
                if significant_changes:
                    for change in significant_changes:
                        pdf.multi_cell(180, 8, change)
                else:
                    pdf.multi_cell(180, 8, "No significant changes (≥5%) in land features detected.")
                
                pdf.ln(10)
                
                # Add poverty status change
                pdf.set_font('Arial', 'B', 12)
                pdf.cell(200, 10, 'Poverty Status Change:', 0, 1, 'L')
                pdf.set_font('Arial', '', 12)
                
                if earlier_status_text != later_status_text:
                    pdf.multi_cell(180, 8, f"The area has changed from {earlier_status_text} in {earlier_year} to {later_status_text} in {later_year}.")
                    
                    # Determine if it's an improvement or decline
                    if "Above" in later_status and "Above" not in earlier_status:
                        pdf.multi_cell(180, 8, "This indicates an improvement in geographical poverty indicators.")
                    elif "Below" in later_status and "Below" not in earlier_status:
                        pdf.multi_cell(180, 8, "This indicates a decline in geographical poverty indicators.")
                    else:
                        pdf.multi_cell(180, 8, "The change in poverty classification is less definitive.")
                else:
                    pdf.multi_cell(180, 8, f"The poverty classification remained {earlier_status_text} from {earlier_year} to {later_year}.")
                
            # Output PDF
            pdf_output = pdf.output(dest='S').encode('latin1', errors='replace')
            st.download_button(
                label="Download Comparison Report",
                data=pdf_output,
                file_name=f"comparison_report_{earlier_year}_{later_year}.pdf",
                mime="application/pdf"
            )
        except Exception as e:
            st.error(f"Error generating PDF report: {str(e)}")
            st.info("Please try again or check console for more information.")