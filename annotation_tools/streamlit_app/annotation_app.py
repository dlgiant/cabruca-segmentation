"""
Streamlit-based Annotation Interface for Cacao Tree Detection
This app provides an interactive interface for annotating cacao trees, shade trees,
and background in satellite/drone imagery.
"""

import base64
import hashlib
import json
import os
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image, ImageDraw

# Configure Streamlit page
st.set_page_config(
    page_title="Cacao Tree Annotation Tool",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Class definitions for annotation
CLASSES = {
    0: {"name": "background", "color": "#808080", "rgb": (128, 128, 128)},
    1: {"name": "cacao", "color": "#00FF00", "rgb": (0, 255, 0)},
    2: {"name": "shade_tree", "color": "#FF0000", "rgb": (255, 0, 0)},
}


class AnnotationSession:
    """Manages the annotation session state"""

    def __init__(self):
        if "annotations" not in st.session_state:
            st.session_state.annotations = {}
        if "current_image" not in st.session_state:
            st.session_state.current_image = None
        if "current_class" not in st.session_state:
            st.session_state.current_class = 1  # Default to cacao
        if "drawing_mode" not in st.session_state:
            st.session_state.drawing_mode = "polygon"
        if "current_polygon" not in st.session_state:
            st.session_state.current_polygon = []
        if "image_list" not in st.session_state:
            st.session_state.image_list = []
        if "current_index" not in st.session_state:
            st.session_state.current_index = 0

    def load_image(self, image_path: str) -> np.ndarray:
        """Load an image from path"""
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def save_annotation(self, image_path: str, annotations: Dict):
        """Save annotations in JSON format compatible with COCO"""
        annotation_dir = Path("data/annotations/streamlit")
        annotation_dir.mkdir(parents=True, exist_ok=True)

        # Generate unique ID for image
        image_id = hashlib.md5(image_path.encode()).hexdigest()[:8]
        timestamp = datetime.now().isoformat()

        annotation_data = {
            "image_path": image_path,
            "image_id": image_id,
            "timestamp": timestamp,
            "annotations": annotations,
            "metadata": {
                "tool": "streamlit_annotation_app",
                "version": "1.0.0",
                "classes": CLASSES,
            },
        }

        # Save to JSON file
        output_file = annotation_dir / f"{Path(image_path).stem}_annotations.json"
        with open(output_file, "w") as f:
            json.dump(annotation_data, f, indent=2)

        return output_file

    def export_to_labelme(self, image_path: str, annotations: List[Dict]) -> Dict:
        """Convert annotations to LabelMe format"""
        labelme_data = {
            "version": "5.0.1",
            "flags": {},
            "shapes": [],
            "imagePath": os.path.basename(image_path),
            "imageData": None,  # We'll keep this None to save space
            "imageHeight": 0,
            "imageWidth": 0,
        }

        # Add shapes
        for ann in annotations:
            shape = {
                "label": CLASSES[ann["class_id"]]["name"],
                "points": ann["points"],
                "group_id": None,
                "shape_type": ann.get("shape_type", "polygon"),
                "flags": {},
            }
            labelme_data["shapes"].append(shape)

        return labelme_data


def draw_annotations_on_image(image: np.ndarray, annotations: List[Dict]) -> np.ndarray:
    """Draw annotations on the image"""
    annotated_image = image.copy()
    overlay = image.copy()

    for ann in annotations:
        class_id = ann["class_id"]
        color = CLASSES[class_id]["rgb"]
        points = np.array(ann["points"], dtype=np.int32)

        if ann.get("shape_type") == "polygon":
            cv2.fillPoly(overlay, [points], color)
            cv2.polylines(annotated_image, [points], True, color, 2)
        elif ann.get("shape_type") == "rectangle":
            pt1 = tuple(points[0])
            pt2 = tuple(points[1])
            cv2.rectangle(overlay, pt1, pt2, color, -1)
            cv2.rectangle(annotated_image, pt1, pt2, color, 2)
        elif ann.get("shape_type") == "circle":
            center = tuple(points[0])
            radius = ann.get("radius", 10)
            cv2.circle(overlay, center, radius, color, -1)
            cv2.circle(annotated_image, center, radius, color, 2)

    # Blend overlay with original image
    alpha = 0.3
    annotated_image = cv2.addWeighted(annotated_image, 1 - alpha, overlay, alpha, 0)

    return annotated_image


def main():
    st.title("🌳 Cacao Tree Annotation Tool")
    st.markdown(
        "### Interactive annotation interface for cacao and shade tree detection"
    )

    # Initialize session
    session = AnnotationSession()
    session.__init__()

    # Sidebar for controls
    with st.sidebar:
        st.header("📁 Data Management")

        # Image upload
        uploaded_files = st.file_uploader(
            "Upload Images",
            type=["png", "jpg", "jpeg", "tif", "tiff"],
            accept_multiple_files=True,
        )

        if uploaded_files:
            st.session_state.image_list = uploaded_files

        # Image selection
        if st.session_state.image_list:
            st.subheader("Image Selection")
            image_names = [f.name for f in st.session_state.image_list]
            selected_image = st.selectbox(
                "Select Image",
                options=range(len(image_names)),
                format_func=lambda x: image_names[x],
                key="image_selector",
            )
            st.session_state.current_index = selected_image

        st.header("🎨 Annotation Controls")

        # Class selection
        selected_class = st.radio(
            "Select Class",
            options=list(CLASSES.keys()),
            format_func=lambda x: f"{CLASSES[x]['name']} ({CLASSES[x]['color']})",
            index=1,
        )
        st.session_state.current_class = selected_class

        # Drawing mode
        drawing_mode = st.selectbox(
            "Drawing Mode", options=["polygon", "rectangle", "circle", "point"], index=0
        )
        st.session_state.drawing_mode = drawing_mode

        # Annotation statistics
        st.header("📊 Statistics")
        if st.session_state.current_image in st.session_state.annotations:
            current_annotations = st.session_state.annotations[
                st.session_state.current_image
            ]

            stats_data = []
            for class_id, class_info in CLASSES.items():
                count = sum(
                    1 for ann in current_annotations if ann["class_id"] == class_id
                )
                stats_data.append(
                    {
                        "Class": class_info["name"],
                        "Count": count,
                        "Color": class_info["color"],
                    }
                )

            df_stats = pd.DataFrame(stats_data)
            st.dataframe(df_stats, hide_index=True)

            # Class distribution pie chart
            if sum(df_stats["Count"]) > 0:
                fig = go.Figure(
                    data=[
                        go.Pie(
                            labels=df_stats["Class"],
                            values=df_stats["Count"],
                            marker_colors=df_stats["Color"],
                        )
                    ]
                )
                fig.update_layout(height=200, margin=dict(t=0, b=0, l=0, r=0))
                st.plotly_chart(fig, use_container_width=True)

        # Export options
        st.header("💾 Export Options")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Save Annotations", type="primary"):
                if (
                    st.session_state.current_image
                    and st.session_state.current_image in st.session_state.annotations
                ):
                    output_file = session.save_annotation(
                        st.session_state.current_image,
                        st.session_state.annotations[st.session_state.current_image],
                    )
                    st.success(f"Saved to {output_file}")

        with col2:
            if st.button("Export to LabelMe"):
                if (
                    st.session_state.current_image
                    and st.session_state.current_image in st.session_state.annotations
                ):
                    labelme_data = session.export_to_labelme(
                        st.session_state.current_image,
                        st.session_state.annotations[st.session_state.current_image],
                    )

                    # Create download link
                    json_str = json.dumps(labelme_data, indent=2)
                    b64 = base64.b64encode(json_str.encode()).decode()
                    href = f'<a href="data:application/json;base64,{b64}" download="{Path(st.session_state.current_image).stem}_labelme.json">Download LabelMe JSON</a>'
                    st.markdown(href, unsafe_allow_html=True)

        if st.button("Clear All Annotations", type="secondary"):
            if st.session_state.current_image in st.session_state.annotations:
                st.session_state.annotations[st.session_state.current_image] = []
                st.rerun()

    # Main annotation area
    if st.session_state.image_list:
        current_file = st.session_state.image_list[st.session_state.current_index]

        # Save uploaded file temporarily
        temp_dir = Path("temp_uploads")
        temp_dir.mkdir(exist_ok=True)
        temp_path = temp_dir / current_file.name

        with open(temp_path, "wb") as f:
            f.write(current_file.getbuffer())

        st.session_state.current_image = str(temp_path)

        # Load and display image
        image = session.load_image(str(temp_path))

        # Initialize annotations for this image if not exists
        if st.session_state.current_image not in st.session_state.annotations:
            st.session_state.annotations[st.session_state.current_image] = []

        # Display annotation interface
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Annotation Canvas")

            # Get current annotations
            current_annotations = st.session_state.annotations[
                st.session_state.current_image
            ]

            # Draw annotations on image
            if current_annotations:
                annotated_image = draw_annotations_on_image(image, current_annotations)
            else:
                annotated_image = image

            # Display image
            st.image(annotated_image, use_column_width=True)

            # Add instructions
            st.info(
                """
            **Instructions:**
            1. Select a class from the sidebar
            2. Choose a drawing mode
            3. Click on the image to add annotations
            4. Use the annotation list to manage existing annotations
            """
            )

        with col2:
            st.subheader("Annotation List")

            if current_annotations:
                for i, ann in enumerate(current_annotations):
                    class_info = CLASSES[ann["class_id"]]

                    with st.expander(
                        f"{class_info['name']} - {ann.get('shape_type', 'polygon')} #{i+1}"
                    ):
                        st.write(f"Class: {class_info['name']}")
                        st.write(f"Type: {ann.get('shape_type', 'polygon')}")
                        st.write(f"Points: {len(ann['points'])}")

                        if st.button(f"Delete", key=f"delete_{i}"):
                            st.session_state.annotations[
                                st.session_state.current_image
                            ].pop(i)
                            st.rerun()
            else:
                st.info("No annotations yet. Start annotating!")

        # Navigation controls
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])

        with col1:
            if st.button("⬅️ Previous", disabled=st.session_state.current_index == 0):
                st.session_state.current_index -= 1
                st.rerun()

        with col2:
            st.markdown(
                f"<center>Image {st.session_state.current_index + 1} of {len(st.session_state.image_list)}</center>",
                unsafe_allow_html=True,
            )

        with col3:
            if st.button(
                "Next ➡️",
                disabled=st.session_state.current_index
                == len(st.session_state.image_list) - 1,
            ):
                st.session_state.current_index += 1
                st.rerun()

    else:
        st.info("👆 Please upload images using the sidebar to begin annotation")

        # Display sample annotation workflow
        st.markdown(
            """
        ### Annotation Workflow
        
        1. **Upload Images**: Use the sidebar to upload satellite/drone images
        2. **Select Class**: Choose between cacao trees, shade trees, or background
        3. **Draw Annotations**: Use polygon, rectangle, or point tools
        4. **Review & Edit**: Check your annotations in the list view
        5. **Export**: Save as JSON or LabelMe format
        
        ### Class Definitions
        
        - 🟢 **Cacao Trees**: Small to medium-sized trees, typically in rows
        - 🔴 **Shade Trees**: Larger canopy trees providing shade
        - ⚫ **Background**: Everything else (ground, buildings, etc.)
        """
        )


if __name__ == "__main__":
    main()
