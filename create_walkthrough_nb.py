import json
import os

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Mask-Guided Visual Captioning Pipeline\n",
            "\n",
            "This notebook demonstrates the pipeline for transitioning from detailed medical explanations to focused visual captions using region of interest (ROI) masks and the Gemini API."
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. The Original Problem: Hallucination from Complex Explanations\n",
            "\n",
            "Initially, the model was trained on complex, clinical explanations. Here are 5 samples of the original images alongside the overly-detailed `exp_ans` targets that led to hallucinations on hard samples:"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "import cv2\n",
            "import os\n",
            "\n",
            "original_csv = r'D:\\FYP_MTL_GI_VQA\\data\\combined\\vqa_exp_v2.csv'\n",
            "IMAGES_DIR = r'D:\\FYP_MTL_GI_VQA\\data\\images'\n",
            "\n",
            "try:\n",
            "    # We randomly grab or just take the first 5 samples\n",
            "    df_orig = pd.read_csv(original_csv).head(5)\n",
            "    \n",
            "    for idx, row in df_orig.iterrows():\n",
            "        img_id = str(row['img_id'])\n",
            "        img_path = os.path.join(IMAGES_DIR, f\"{img_id}.jpg\")\n",
            "        \n",
            "        if os.path.exists(img_path):\n",
            "            img = cv2.imread(img_path)\n",
            "            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\n",
            "            \n",
            "            fig, ax = plt.subplots(1, 1, figsize=(4, 3))\n",
            "            ax.imshow(img)\n",
            "            ax.axis('off')\n",
            "            ax.set_title(f\"Image ID: {img_id}\")\n",
            "            plt.show()\n",
            "            \n",
            "            print(\"Detailed Explanation (exp_ans):\")\n",
            "            print(row['exp_ans'])\n",
            "            print(\"-\" * 80)\n",
            "        else:\n",
            "            print(f\"Image {img_id}.jpg not found.\")\n",
            "            \n",
            "except Exception as e:\n",
            "    print(f\"Could not load the original CSV file: {e}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Visualizing Masks Overlaid on Images\n",
            "\n",
            "To solve the hallucination, we pivot to grounded visual descriptions. We load the original images and their corresponding masks (e.g., polyp, instrument, GradCAM), and overlay contours and bounding boxes. This explicitly forces the API vision model to look at the marked area."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import glob\n",
            "\n",
            "DATA_DIR = r'D:\\FYP_MTL_GI_VQA\\data'\n",
            "masks = glob.glob(os.path.join(DATA_DIR, \"polyp_masks\", \"*.jpg\"))\n",
            "\n",
            "if masks:\n",
            "    MASK_PATH = masks[0]\n",
            "    img_id = os.path.basename(MASK_PATH).split('.')[0]\n",
            "    IMG_PATH = os.path.join(IMAGES_DIR, f\"{img_id}.jpg\")\n",
            "\n",
            "    if os.path.exists(IMG_PATH) and os.path.exists(MASK_PATH):\n",
            "        img = cv2.imread(IMG_PATH)\n",
            "        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\n",
            "        \n",
            "        mask = cv2.imread(MASK_PATH, cv2.IMREAD_GRAYSCALE)\n",
            "        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)\n",
            "        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)\n",
            "        \n",
            "        overlay = img.copy()\n",
            "        for c in contours:\n",
            "            cv2.drawContours(overlay, [c], -1, (0, 255, 0), thickness=3) # Green contour\n",
            "            x, y, w, h = cv2.boundingRect(c)\n",
            "            cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 0, 0), thickness=3) # Red bounding box\n",
            "            \n",
            "        fig, ax = plt.subplots(1, 3, figsize=(15, 5))\n",
            "        ax[0].imshow(img)\n",
            "        ax[0].set_title(f'Original Image ({img_id})')\n",
            "        ax[0].axis('off')\n",
            "        \n",
            "        ax[1].imshow(binary_mask, cmap='gray')\n",
            "        ax[1].set_title('Mask')\n",
            "        ax[1].axis('off')\n",
            "        \n",
            "        ax[2].imshow(overlay)\n",
            "        ax[2].set_title('Overlay (Fed to Model)')\n",
            "        ax[2].axis('off')\n",
            "        plt.show()\n",
            "    else:\n",
            "        print(\"Image or Mask not found.\")\n",
            "else:\n",
            "    print(\"No masks found in the expected directory.\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Constructing the Prompt\n",
            "\n",
            "The prompt is heavily constrained. It provides the ROI type and specific guidance directly telling the model (Gemma 3 27B) what to focus on based on the category of the highlighted region."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "ROI_TAXONOMY = {\n",
            "    \"polyp\": (\"Polyp\", \"polyp\"),\n",
            "    \"z-line\": (\"Z-line\", \"landmark\"),\n",
            "    \"instrument\": (\"Instrument\", \"instrument\")\n",
            "}\n",
            "\n",
            "def build_user_prompt(roi_type):\n",
            "    display_name, category = ROI_TAXONOMY.get(roi_type, (\"Unknown\", \"unknown\"))\n",
            "    \n",
            "    hints = {\n",
            "        \"polyp\": \"Focus on the shape of the marked structure \u2014 whether it is raised, flat, or has a stalk \u2014 and describe its surface color and texture.\",\n",
            "        \"landmark\": \"Focus on the visible boundary or transition zone. Describe any color change or difference in surface texture.\",\n",
            "        \"instrument\": \"Focus on the physical appearance of the object inside the marked region. Describe its shape, color, and surface material.\"\n",
            "    }\n",
            "    \n",
            "    system_prompt = \"\"\"You are a visual description assistant for gastrointestinal endoscopy images.\n",
            "You will be given an endoscopy image with a bounding box and contour drawn around a region of interest (ROI).\n",
            "Write a short visual description of what is visible inside the marked region.\n",
            "- Maximum 2 sentences.\n",
            "- Use plain language \u2014 no medical or clinical terms.\n",
            "- CRITICAL: Start directly with the visual description.\"\"\"\n",
            "\n",
            "    user_prompt = f\"{system_prompt}\\n\\nROI type: {category.capitalize()}\\nROI name: {display_name}\\n\\n\"\n",
            "    user_prompt += \"The bounding box and contour in the image mark the region of interest. \"\n",
            "    user_prompt += \"Provide a direct visual description in 1\u20132 sentences without introductory filler.\\n\\n\"\n",
            "    user_prompt += hints.get(category, \"\")\n",
            "    \n",
            "    return user_prompt\n",
            "\n",
            "print(\"-- Example Prompt sent to Gemini for a Polyp --\\n\")\n",
            "print(build_user_prompt('polyp'))"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Viewing the Generated Samples\n",
            "\n",
            "After running the generation script, the short captions generated by Gemini are saved alongside the corresponding mask logic. Here are 5 visual pairs showing the updated visual captions generated."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "csv_path = r'D:\\FYP_MTL_GI_VQA\\generated_captions.csv'\n",
            "\n",
            "try:\n",
            "    df_gen = pd.read_csv(csv_path).head(5)\n",
            "    print(f\"Displaying 5 Generated Short Captions based on ROIs:\\n\")\n",
            "    \n",
            "    for idx, row in df_gen.iterrows():\n",
            "        img_id = str(row['img_id'])\n",
            "        roi_type = row['roi_type']\n",
            "        vis_des = row['vis_des']\n",
            "        \n",
            "        img_path = os.path.join(IMAGES_DIR, f\"{img_id}.jpg\")\n",
            "        \n",
            "        if os.path.exists(img_path):\n",
            "            img = cv2.imread(img_path)\n",
            "            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\n",
            "            \n",
            "            fig, ax = plt.subplots(1, 1, figsize=(4, 3))\n",
            "            ax.imshow(img)\n",
            "            ax.axis('off')\n",
            "            ax.set_title(f\"Image ID: {img_id} | ROI: {roi_type}\")\n",
            "            plt.show()\n",
            "            \n",
            "            print(f\"Visual Description (vis_des):\\n{vis_des}\")\n",
            "            print(\"-\" * 80)\n",
            "        else:\n",
            "            print(f\"Image {img_id}.jpg not found.\")\n",
            "            \n",
            "except FileNotFoundError:\n",
            "    print(f\"Unable to find file: {csv_path}\\nPlease run the generation script first.\")"
        ]
    }
]

notebook_data = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.x"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open(r'd:\FYP_MTL_GI_VQA\Solution_Walkthrough.ipynb', 'w') as f:
    json.dump(notebook_data, f, indent=1)

print("Notebook generated successfully!")
