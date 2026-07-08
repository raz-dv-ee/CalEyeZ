# SHENKAR - ENGINEERING. DESIGN. ART.
**The Pernick Faculty of Engineering**

# CalEyeZ
## Electrical and Electronic Engineering B.Sc. Final Project

### FRS - Functional Requirement Specification

**Authors:** Roi Zur  
Raz Dvora  

**Supervisors:** Dr. Gabriela Dorfman Furman  
Dr. Zeev Weissman  

**Approved by:** Zeev, GA  
**Date:** 24/12/2025

---

## Table of Contents
1. Overview
2. Problem Description and Market Overview
   2.1. Available solutions in the market
   2.2. Project Goals
3. The Approach
4. Use Case Scenarios
   4.1. Description of typical users (actors) and their goals
   4.2. Description of use case scenario
5. Functional Requirements
   5.1. Unit 1 requirements
   5.2. Unit 2 requirements
   5.3. User Interface
6. Non-Functional Requirements
   6.1. Performance
   6.2. Costs
   6.3. Energy
   6.4. Environmental
   6.5. Health & Safety
   6.6. Compliance with standards and regulations
   6.7. Operational
   6.8. Usability
7. System Flows
8. Non Goals & Restrictions
9. Open Issues & Risk Management

---

## 1. Overview

**Problem Description:** Tracking nutrition manually is tedious and inaccurate, causing most users to give up. Existing AI solutions face two main hurdles: they struggle to learn new, local cuisines without "forgetting" standard foods, and they cannot accurately guess food weight from a simple photo due to differences in food density (like the difference between a fluffy bun and a dense steak).

**Solution:** CalEyez is an automated system that solves these problems using a "Smart Ensemble" approach. Instead of relying on a single AI brain, our system uses two specialized experts: one for standard international foods and one for specific local dishes. A smart "Router" acts as a traffic controller, analyzing each image and sending it to the expert most likely to identify it correctly. For portion sizing, rather than guessing volume, the system is designed to read the exact weight directly from a digital scale's display.

**Innovation:** The key innovation is the system's modular architecture. By separating general knowledge from specific local knowledge, we achieved a high system accuracy of 80.30%, proving that a team of specialized AI models outperforms a single general one. Additionally, shifting from geometric volume estimation to reading the scale's digital display transforms the system from a "guessing tool" into a precise nutritional instrument.

## 2. Problem Description and Market Overview

The development of an automated nutritional analysis system faces two distinct and significant engineering hurdles:

**Problem 1: The "WHAT?" Challenge (Food Recognition)**
The first major obstacle is accurately identifying the specific food item present in an image.
* **The Monolithic Model Failure:** Training a single Artificial Intelligence model to recognize a vast array of food items (150+ classes) is inherently unstable. Standard datasets (like Food-101) cover general international foods but lack specific local cuisines (e.g., Israeli dishes).
* **Catastrophic Forgetting:** When attempting to "teach" an existing model these new, local foods, the model frequently suffers from "catastrophic forgetting." As it adjusts its weights to learn the new classes (like Jachnun), it drastically loses accuracy on the original classes (like Steak), making it nearly impossible to create a single, unified classifier that performs well on both datasets simultaneously.
* **Visual Similarity:** Many food items have extremely high inter-class similarity (e.g., distinguishing between Schnitzel and Crispy Chicken, or different types of nuts). Standard models struggle to differentiate these fine-grained details without specialized training.

**Problem 2: The "HOW MUCH?" Challenge (Weight Estimation)**
The second, and arguably more complex challenge, is determining the quantity (weight) of the food from a 2D image to calculate calories.
* **The Failure of Volume Estimation:** Attempting to calculate weight by first estimating volume is fundamentally flawed. Computer vision techniques for depth estimation (like MiDaS) or geometric shadow analysis are unreliable for small, irregular objects like food. They are highly sensitive to lighting conditions and camera angles, leading to massive errors in volume calculation.
* **The Density Variable:** Even if a system could calculate volume perfectly (e.g., "This pile of rice is 200cm³"), converting that volume to weight is prone to high error margins because food density is inconsistent. A "fluffy" pile of mashed potatoes weighs significantly less than a dense scoop of the same size. Therefore, any visual method relying on volume mathematics cannot provide the precision required for nutritional tracking.

### 2.1 Available solutions in the market

**1. MyFitnessPal**
MyFitnessPal is currently the market leader in nutritional tracking. It relies primarily on a massive, crowdsourced database of food items.
* **Method:** Users predominantly log food by manually searching for items by text or scanning barcodes on packaging. While it has introduced a "Meal Scan" feature, it is often gated behind premium subscriptions or serves as a secondary input method.
* **Limitations:** The system is heavily dependent on manual user input for portion sizes. It does not automatically calculate weight from an image; the user must weigh the food separately and type in the value (e.g., "150g"), leaving the process prone to human error and estimation fatigue.

**2. FoodVisor**
FoodVisor is a direct competitor that focuses on AI-based visual recognition. It attempts to identify food and estimate serving sizes from photos.
* **Method:** The app uses deep learning to segment food items in an image and attempts to estimate volume (3D analysis) to calculate calories.
* **Limitations:** While innovative, the volume estimation is frequently inaccurate due to the "density problem" described in the problem statement. Users often have to manually adjust the bounding boxes or correct the estimated portion size because the camera cannot distinguish between dense and light foods reliably. Furthermore, its classifier is trained on global datasets and often struggles with specific local or complex mixed cuisines.

### 2.2 Project Goals

The primary goal of the CalEyez project is to develop a fully automated, end-to-end system for nutritional analysis that overcomes the reliability issues of current market solutions. Specific objectives include:
* **High-Accuracy Classification:** To develop a computer vision system capable of identifying 166 diverse food items with a validated system accuracy of at least 80%.
* **Modular Scalability:** To validate a Router-based Ensemble architecture that allows for the integration of specific local cuisines (e.g., Israeli food) without degrading the performance of the base model (solving "catastrophic forgetting").
* **Precise Weight Extraction:** To implement a robust OCR (Optical Character Recognition) pipeline that reads weight directly from a digital scale display, providing a deterministic and accurate alternative to error-prone geometric volume estimation.
* **End-to-End Automation:** To create a seamless software pipeline that accepts a single raw image as input and returns a complete nutritional breakdown (calories, proteins, fats) by integrating classification results with external databases (USDA), thereby minimizing user friction.

## 3. The Approach

**Addressing the Needs:** Our approach directly tackles the core challenges of scalability and accuracy by implementing an Ensemble Learning architecture managed by a Smart XGBoost Router. This architecture addresses the "catastrophic forgetting" problem by decoupling general food classification from specific local cuisines, allowing the system to maintain 80.30% accuracy across 158 classes without retraining a massive monolithic model for every new addition.

The router intelligently directs images to the most appropriate expert model (achieving 99.3% accuracy on routing "Old" data), ensuring high precision even for visually similar items. Additionally, the project tackles the weight estimation challenge by pivoting from unreliable geometric/volume estimation to a robust OCR-based approach, which directly reads the digital scale display, eliminating errors caused by food density variations and lighting conditions.

Compared to existing market solutions like MyFitnessPal (manual entry) and FoodVisor (prone-to-error volume estimation), CalEyez offers distinct advantages. Its primary benefit is automation with precision: it removes the user friction of manual logging while providing a more deterministic and reliable weight measurement via OCR. Furthermore, the modular router design offers superior adaptability, allowing developers to easily plug in new "expert" models for different cuisines (e.g., Asian, Indian) without degrading the performance of the core system, a flexibility that monolithic commercial competitors lack. This results in a scalable, high-accuracy tool that directly addresses the pain points of user adherence and data reliability.

## 4. Use Case Scenarios

### 4.1. Description of typical users (actors) and their goals

**The End User (Health-Conscious Individual / Dieter)**
* **Goal:** To log a meal effortlessly by taking a single photo, avoiding the need to manually search for food items or type in weights.
* **Goal:** To receive an accurate identification of the food, even for specific local cuisines (e.g., Jachnun, Sufganiyah) that standard apps often miss.
* **Goal:** To obtain a precise nutritional breakdown (Calories, Protein, Carbs, Fats) based on the exact weight displayed on their digital scale.

**The Dietitian / Nutritionist (Professional User)**
* **Goal:** To monitor clients' dietary intake using objective, image-based data rather than relying on subjective self-reporting.
* **Goal:** To automate the tedious process of calculating macronutrients for complex or home-cooked meals.

**The System Administrator / Data Engineer**
* **Goal:** To scale the system by adding new food classes to the "New Model" (specialized expert) without needing to retrain the massive "Old Model" or degrade its performance.
* **Goal:** To retrain the XGBoost Router to maintain high routing accuracy (>93%) as new data is added to the system.

### 4.2. Description of use case scenario

**PART 1: Food Identification**
* User Takes Photo -> Input Image -> XGBoost Router
* Router splits to either: OLD Model (150 classes) OR NEW Model (8 classes)
* Result: Food Name

**PART 2: Weight Extraction**
* Detect Display -> Crop Screen -> OCR Model -> Result: Weight (g)

**Data Fusion:**
* Food Item + Weight -> USDA API (External Service) -> Calc Nutrients -> Final Report

## 5. Functional Requirements

### 5.1 Unit 1 Requirements (Image Classification Engine)
* This unit is responsible for the core AI logic, including image preprocessing, model inference, and routing decisions.
* The system will accept digital images in standard formats (JPG, PNG, BMP) and automatically resize them to a resolution of 224x224 pixels for model compatibility.
* The system will run inference using two distinct YOLOv8-based models: an 'Old Model' trained on 150+ classes and a 'New Model' trained on 8 specific local classes.
* The system will implement an XGBoost router that accepts feature vectors (confidence scores, entropy, margins) from both models and outputs a binary classification decision (0 or 1) to select the active model.
* The system will achieve an aggregate top-1 classification accuracy of at least 80% when evaluated against the project's validation dataset (16,720 images).
* The system will complete the entire classification pipeline (preprocessing -> inference -> routing) in under 2.0 seconds on a PC equipped with a CUDA-enabled GPU (e.g., NVIDIA RTX 3060).

### 5.2 Unit 2 requirements (Weight Extraction & Data Fusion)
* This unit handles the "How Much?" challenge via OCR and the integration with external nutritional databases.
* The system will identify the Region of Interest (ROI) containing the digital scale's 7-segment display within the input image.
* The system will utilize an OCR engine (e.g., EasyOCR) to extract a numeric string from the identified ROI and convert it into a floating-point value representing grams.
* The system will send an HTTP GET request to the USDA FoodData Central API containing the identified food string (e.g., "Schnitzel") to retrieve nutritional values per 100g.
* The system will calculate the final nutritional values (Calories, Protein, Fat, Carbs) by applying the formula: `Value_per_100g * (DetectedWeight / 100)`.
* The system will return a default "Not Found" error message if the API response time exceeds 5000ms (5 seconds).

### 5.3. User Interface
These requirements define the interaction between the user and the Tkinter GUI.
* The User Interface will include a "Load Image" button that opens the operating system's native file explorer dialog to select an image.
* The User Interface will display the selected image in a central canvas area, automatically scaled to fit within the window dimensions while maintaining the aspect ratio.
* The User Interface will display the classification results in a dedicated text panel, explicitly showing: the predicted Class Name, the Confidence Score (%), and the Router's Decision ("Winner: OLD" or "Winner: NEW").
* The User Interface will allow the user to toggle "Grad-CAM" visualization via a distinct button, overlaying a heatmap on the original image to indicate the model's focus area.
* The system will display the final calculated nutritional report (Calories, Macros) in a structured table format on the main dashboard after processing is complete.

## 6. Non-Functional Requirements

### 6.1. Performance
* The system shall process a single image (from loading to final nutritional output) in less than 3 seconds on a standard PC with an NVIDIA RTX 3060 GPU.
* The system shall utilize no more than 6GB of GPU VRAM during inference to ensure compatibility with mid-range consumer hardware.
* The XGBoost Router shall make a routing decision in less than 100 milliseconds.

### 6.2. Costs
* The software implementation shall rely exclusively on open-source libraries (Python, PyTorch, Ultralytics) to ensure zero licensing costs for the runtime environment.
* The external data retrieval shall utilize the free tier of the USDA FoodData Central API.

### 6.3. Energy
* The system shall implement a "lazy loading" or cleanup mechanism to release GPU resources when the system is idle for more than 5 minutes, reducing power consumption.
* The application logic shall be optimized to prevent CPU usage from exceeding 10% during idle states (when no image is being processed).

### 6.4. Environmental
* The system eliminates the need for paper-based food diaries, digitizing the entire logging process.
* The software is designed to run on existing, general-purpose consumer hardware (PCs/Laptops), negating the environmental impact of manufacturing proprietary scanning devices.

### 6.5. Health & Safety
* Disclaimer Requirement: The user interface shall prominently display a disclaimer stating that nutritional values are estimates based on computer vision and should not be used for critical medical decisions (e.g., insulin dosing for diabetics).
* Hygiene: The system shall operate using a non-contact method (photography), ensuring no physical contact with the food is required for measurement.

### 6.6. Compliance with standards and regulations
* Licensing: The system shall comply with the AGPL-3.0 license requirements inherent to the Ultralytics YOLOv8 library used for object detection.
* Data Standards: The nutritional data output shall conform to the USDA FoodData Central data schema for macronutrient reporting.

### 6.7. Operational
* OS Compatibility: The system shall be fully operational on Windows 10 and Windows 11 (64-bit) environments.
* Hardware Dependency: The system requires a GPU capable of PyTorch acceleration (supporting NVIDIA CUDA or AMD ROCm/DirectML) to ensure inference times remain under 3 seconds.
* Lighting Conditions: The system's OCR component requires input images to have a minimum illumination of 300 lux (standard office/kitchen lighting) to ensure legible reading of the scale display.

### 6.8. Usability
* Learning Curve: The system's Graphical User Interface (GUI) shall be designed with a single-window workflow so that a user with basic computer literacy can learn 80% of the system's functionality (loading an image, running the analysis, and interpreting results) within 15 minutes of first use, significantly exceeding the 2-hour requirement.
* Function Count Constraints: The user interface shall contain no more than 10 distinct user-facing functions to ensure simplicity. The defined functions are:
    1.  Open Image
    2.  Zoom In
    3.  Zoom Out
    4.  Reset Zoom
    5.  Toggle Grad-CAM (Visualization)
    6.  Select "Old" Model
    7.  Select "New" Model
    8.  Select Router
    9.  Set Confidence Threshold
    10. View Nutritional Report

## 7. System Flows
* **Start:** User Capture -> Input: Image Data -> Process: Resize (224x224)
* **Branch 1 (Weight Extraction):** Process: Detect Screen ROI -> Process: Crop Display -> Process: Run OCR -> Data: Weight (g)
* **Branch 2 (Food Identification):** Process: Run OLD Model / Process: Run NEW Model -> Decision: Check Scores -> Select Best Model -> Data: Food Name -> Database: USDA API -> Return Macros/100g
* **Merge:** Data: Weight (g) + Return Macros/100g -> Process: Calculate Nutrients -> Output: Final Report -> **End**

## 8. Non-Goals & Restrictions

**Non-Goals (What the project will NOT do)**
* **Volumetric Estimation:** The system will not attempt to calculate food weight based on 3D depth estimation, shadow analysis, or geometric volume formulas. These methods were evaluated and rejected due to the "density problem" and high error rates.
* **Mobile Application:** The current scope is limited to a desktop-based prototype (PC). Developing native mobile applications (iOS/Android) is out of the scope of this final project.
* **Medical Diagnosis:** The system provides nutritional data for informational purposes only. It is not a medical device and does not provide dietary prescriptions or medical advice for conditions like diabetes or obesity.
* **Multi-Food Plate Analysis:** The current system is optimized to identify and analyze a single dominant food item per image. It is not designed to segment and calculate mixed plates with multiple overlapping food types simultaneously.
* **Real-Time Video Processing:** The system is designed for static image analysis ("Snap and Process"), not for continuous real-time video stream classification.

**8.1. Restrictions (System Limitations)**
* **Dataset Limitation:** The system can only identify food items belonging to the 166 specific classes it was trained on (158 standard + 8 local). Any food item outside this list will likely be misclassified as the nearest visual match.
* **Hardware Dependency:** To achieve the required inference speed (<3 seconds), the system requires a PC with a dedicated GPU (NVIDIA RTX series recommended) supporting CUDA acceleration. Running on CPU only may result in significant latency.
* **Input Constraints for Weight:** For the weight extraction feature to function, the input image must include a clear, unobstructed view of a digital scale's 7-segment display.
* **Connectivity:** The system requires an active internet connection to query the USDA API for nutritional values; it cannot function in a fully offline mode for the final report generation.
* **Lighting Conditions:** Extreme lighting conditions (too dark or significant glare on the scale's screen) may cause the OCR or classification modules to fail.

## 9. Open Issues & Risk Management

| Item | Detail | Status | Responsibility | Proposed resolution | Estimated resolution |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Weight Extraction (OCR) | The transition from volume estimation to OCR is theoretically sound, but reading 7-segment displays in variable lighting (glare/reflection) is prone to errors. | Open | Raz & Roi | 1. Implement strict ROI filtering (finding the black screen box first).<br>2. Use specific OCR libraries tuned for digits (e.g., EasyOCR).<br>3. Fallback: Allow manual user correction if OCR fails. | 15/05/2026 |
| Router "New Data" Accuracy | While the system accuracy is 80%, the Router still struggles with the "New" model classes (37% error rate on new classes due to small dataset). | In Progress | Raz | 1. Collect more images specifically for the 8 "New" classes to balance the router's training set.<br>2. Retrain Router v4. | 29/2/26 |
| USDA API Limitations | Dependence on an external API for nutritional data. The API might change formats, be down, or have rate limits. | Open | Raz | 1. Cache common food results locally (JSON database).<br>2. Implement error handling to show "Service Unavailable" gracefully. | 30/04/2026 |
