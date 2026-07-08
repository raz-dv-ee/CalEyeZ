# SHENKAR - ENGINEERING. DESIGN. ART.
**The Pernick Faculty of Engineering**

# CalEyeZ
## Electrical & Electronic Engineering B.Sc. Final Project

### System Design Document

**Authors:** Raz Dvora  
Roi Tzur  

**Supervisors:** Dr. Gabriela Dorfman Furman  
Dr. Zeev Weissman  

**Date:** 05/02/2026

---

## Table of Contents
1. Introduction
   a. System Overview
   b. Purpose and Scope
   c. Definitions and Acronyms
   d. Use Case
   e. Constraints
2. System Architecture
3. Literature Survey
   a. Problems and Solutions Survey
   b. Discussion and Conclusion
4. Technological Survey
   a. Technological alternatives Survey
   b. Discussion and Conclusion
5. Design
   a. System Design
   b. Description of Components
   c. GUI
   d. BOM-Bill of materials
6. Risk Management
7. Verification
8. Project Management
   a. Schedule
   b. Team Roles

---

## 1. Introduction

### a. System Overview
CalEyeZ is an integrated IoT solution designed to automate the process of nutritional tracking. The system utilizes a "Sensor Fusion" approach, combining Computer Vision (CV) for food classification with a reverse-engineered Bluetooth Low Energy (BLE) interface for precise weight acquisition. The core of the system is a Python-based desktop application that runs on a standard PC. It employs Hybrid AI Architecture (Global + Local Models) to identify food items specifically optimized for Israeli cuisine, and cross-references the detected class and weight with the USDA nutritional database to provide real-time caloric feedback to the user via a modern GUI.

### b. Purpose and Scope
**Purpose:** Manual calorie counting is tedious, prone to human error, and often leads to user abandonment. The purpose of CalEyeZ is to remove the friction from this process by automating the data entry. By simply placing a plate on the scale, the user receives an immediate breakdown of Calories, Protein, Fats, and Carbs.

**In Scope:**
* Real-time detection of food items using YOLOv8 (Object Detection).
* Wireless weight data acquisition from a modified digital scale (BLE Protocol).
* Algorithmic retrieval of nutritional values (USDA API with heuristic filtering).
* Support for both international staples (e.g., Apple, Bread) and local Israeli dishes (e.g., Jachnun, Sabich).

**Out of Scope:**
* Medical advice or dietary prescriptions.
* Mobile application interface (currently limited to Desktop/Workstation).
* Detection of mixed/mashed foods (e.g., smoothies) where ingredients are indistinguishable.

### c. Definitions and Acronyms

| Acronym | Definition |
| :--- | :--- |
| **BLE** | Bluetooth Low Energy: A wireless personal area network technology used for transmitting weight data from the scale to the PC. |
| **YOLO** | You Only Look Once: A real-time object detection algorithm used for the computer vision engine to identify food items. |
| **GUI** | Graphical User Interface: The visual dashboard (built with CustomTkinter) allowing user interaction and data visualization. |
| **GATT** | Generic Attribute Profile: The Bluetooth protocol layer used to decode the scale's raw data packets into readable weight values. |
| **OCR** | Optical Character Recognition: A technology initially researched for reading the scale's 7-segment display (deprecated in favor of BLE due to instability). |
| **USDA** | United States Department of Agriculture: The external source database used for retrieving caloric and nutritional values. |
| **V&V** | Verification and Validation: The engineering process of checking that the system meets specifications and fulfills its intended purpose. |

### d. Use Case
**Scenario: "Logging a Lunch Meal (Router Logic)"**
1.  **Setup:** The user launches the CalEyeZ application and ensures the Swan BLE Scale is active.
2.  **Placement:** The user places a complex food item (e.g., a "Bourekas") on the scale.
3.  **Acquisition:**
    * The Camera captures the visual features of the object.
    * The BLE Module intercepts the weight packet (e.g., 120g) from the scale.
4.  **Processing:**
    * The user clicks "Analyze".
    * The Smart Router evaluates image features against both the Global and Local models.
    * Detecting high confidence in the specific Israeli dataset features, the Router prioritizes the 'Local Cuisine Model' (overriding the Global baseline) and correctly identifies the item as "Cheese Bourekas".
    * The system then applies the heuristic filter to query the nutritional database.
5.  **Output:** The screen displays: "Cheese Bourekas (120g) - 350 kcal" along with a macronutrient breakdown chart.
6.  **Logging:** The data is saved to the user's daily history log.

### e. Constraints
* **Lighting Conditions:** The Computer Vision module requires adequate ambient lighting to distinguish between visually similar foods (e.g., Red Apple vs. Peach).
* **Hardware Dependency:** The system requires a PC with a dedicated GPU (NVIDIA CUDA) for reasonable inference latency (<1 sec) and a specific BLE-enabled scale chipset.
* **Database Limitations:** The accuracy of the nutritional output is bounded by the available entries in the USDA database; obscure local foods may require manual data entry.
* **Connectivity:** The host machine must have a Bluetooth 4.0+ adapter to communicate with the scale.

## 2. System Architecture
CalEyeZ is designed as a standalone Edge-Computing IoT solution, prioritizing user privacy, low latency, and offline capabilities. The architecture shifts the heavy computational load from the cloud to the local edge device, utilizing custom-trained neural networks for real-time food recognition.

* **Desktop Application (Edge Device):** User triggers "Analyze" on the **GUI/Dashboard** (Tkinter Interface).
* **Inputs:** **Swan Scale** sends HEX Data to **BLE Handler** which sends Weight (g) to the **Fusion Logic Controller**. **Camera** sends RGB Stream to **Video Capture** which sends a Frame to both the **Fusion Logic Controller** and the **AI Inference (YOLOv8 Local)**.
* **Processing:** The **AI Inference** sends a Label to the **Fusion Logic Controller**. The Fusion Logic Controller queries the **USDA API** (Food Name) and receives a Nutrients JSON back.
* **Outputs:** The Fusion Logic sends Results to the **GUI** and saves records to the **Log Manager (CSV Storage)**.

## 3. Literature Survey

### a. Problems and Solutions Survey
Current nutritional tracking methodologies primarily rely on manual data acquisition, which introduces significant latency and quantization errors. From a systems engineering perspective, existing solutions suffer from three fundamental flaws that the CalEyeZ system aims to resolve:

* **Discontinuity of Data Acquisition (The "Air Gap"):**
    * **The Engineering Problem:** Existing "smart" scales function merely as isolated transducers. They convert physical load into digital signals (ADC) but lack semantic understanding of the load. There is no direct data link between the measurement device (the scale) and the processing unit (the logging application).
    * **Impact:** This forces the user to act as a "manual bridge," reading the display and typing the value into an application. This introduces high user friction and transcription errors, leading to low long-term user retention.
* **Human Estimation Variance (Measurement Error):**
    * **The Engineering Problem:** Users often rely on visual estimation (e.g., "one medium apple") rather than precise gravimetric measurement.
    * **Impact:** Visual volume estimation by humans is notoriously inaccurate, with variances often exceeding ±30%. Without a closed-loop feedback system that enforces precise weighing, the input data for any nutritional algorithm remains unreliable ("Garbage In, Garbage Out").
* **Classification Latency in Unstructured Environments:**
    * **The Engineering Problem:** While barcode scanners are effective for packaged goods, fresh produce (fruits, vegetables, cooked meals) lacks digital identifiers. Previous Computer Vision generations (based on simple color histograms) failed to achieve sufficient Mean Average Precision (mAP) to differentiate between visually similar foods.
    * **Impact:** Users are forced to manually search databases for generic items, further increasing the time-to-log.
* **Cloud Dependency & Latency:**
    * **The Engineering Problem:** Many modern IoT kitchen appliances rely on cloud offloading for processing.
    * **Impact:** This architecture creates dependency on internet connectivity and introduces network latency, which is unacceptable for real-time user interface feedback. A robust engineering solution requires edge-computing capabilities to perform inference locally.

### b. Discussion and Conclusion

**Comparison of Technological Alternative:**
The following table presents a comparative analysis between the proposed CalEyeZ solution and existing market technologies, evaluating them based on Automation Level, Precision, and User Friction.

| Technological Alternative | Engineering Approach | Pros (Advantages) | Cons (Disadvantages) |
| :--- | :--- | :--- | :--- |
| **Manual Logging Apps** (e.g. MyFitnessPal) | Database Lookup & Manual Entry | - High database granularity<br>- Zero hardware cost<br>- Mature ecosystem | - High Friction: Requires manual search & typing<br>- Low Accuracy: Relies on subjective user estimation<br>- No hardware integration |
| **Barcode Scanners** | Optical Recognition (1D/2D Codes) | - Deterministic: 100% Identification accuracy<br>- Fast processing (O(1) complexity) | - Limited Scope: Ineffective for fresh food (fruits/veg)<br>- Requires package presence<br>- User must locate the code |
| **Standard Smart Scales** (e.g. Renpho/Garmin) | BLE Connectivity + Manual Selection | - High Precision: Accurate gravimetric measurement<br>- Wireless data transfer | - "Dumb" Sensor: No object recognition<br>- User must manually select food type in app<br>- Disconnected workflow |
| **CalEyeZ System** (Proposed Solution) | Sensor Fusion (Weight + Vision) | - Fully Automated Loop: Place & Log (Zero friction)<br>- High Precision: Gravimetric + Visual<br>- Edge Capability: Offline & Fast | - Dependent on camera lighting conditions<br>- Requires initial setup (Hardware)<br>- Computationally intensive (GPU/NPU) |

**Conclusion:**
Based on the analysis above, the CalEyeZ System offers the optimal engineering compromise. While manual apps are accessible and barcode scanners are accurate for packaged goods, neither solves the problem of tracking fresh, unstructured food with high precision. By implementing Sensor Fusion architecture, combining the gravimetric accuracy of a digital scale with the semantic understanding of a YOLOv8 Neural Network, the CalEyez system eliminates the "Human in the Loop" for data entry. This results in a system that provides the precision of a laboratory scale with the ease of use of a consumer appliance, effectively solving the latency and accuracy problems identified in the survey.

## 4. Technological Survey

### a. Technological alternatives Survey
In the design phase of the CalEyeZ system, several technological approaches were evaluated to meet the core requirements of low latency, high accuracy, and user privacy. The following subsections present the trade-off analysis for the key system modules.

### b. Discussion and Conclusion
Following the comparative analysis of available technologies for object detection, wireless communication, and processing architecture, we have selected the following technology stack for the CalEyez system:
1.  **Inference Engine:** We have chosen YOLOv8 over two-stage detectors (like Faster R-CNN) due to its superior real-time performance (FPS) and SOTA accuracy, which are critical for instant user feedback.
2.  **Connectivity:** We have selected Bluetooth Low Energy (BLE) instead of Wi-Fi or Classic Bluetooth. This choice optimizes the hardware's power consumption, significantly extending battery life while maintaining sufficient bandwidth for data packets.
3.  **Processing Architecture:** We have opted for Local Edge Computing rather than Cloud APIs. This strategic decision eliminates operational costs (OpEx), ensures user privacy, and removes latency caused by network dependency.

**Final Determination:** The chosen combination of YOLOv8 + BLE + Edge Computing represents the optimal engineering balance between performance, cost-efficiency, and user experience, enabling the CalEyeZ system to function as a robust, standalone IoT product.

## 5. Design

### a. System Design
The CalEyeZ system is designed as a modular Edge-IoT workstation that integrates physical instrumentation with artificial intelligence. The system logic follows strictly event-driven architecture, orchestrated by a central control unit running on the host machine. The operational flow consists of three synchronized stages:

1.  **Data Acquisition (Sensory Layer):** The system continuously monitors the physical environment through two parallel channels. The Swan Scale broadcasts weight telemetry packets via Bluetooth Low Energy (BLE) at a frequency of 10Hz, providing a real-time stream of the load placed on the sensor. Simultaneously, the Optical Sensor (Camera) maintains a live video feed, buffering frames in memory for immediate access.
2.  **Sensor Fusion & Inference (Processing Layer):** Upon user triggering (via the GUI "Analyze" command), the Fusion Logic freezes the current state. It captures the stabilized weight value and the corresponding video frame. This frame is passed to the local YOLOv8 Inference Engine, which performs object detection and classification. The system then queries the nutritional database to retrieve the macronutrient factors (per 100g) associated with the identified class label.
3.  **Synthesis & Output (Application Layer):** The core algorithm calculates the final nutritional values by applying the gravimetric data to the nutritional factors (value_total = weight_g * factor_per_g). The results are visualized on the dashboard, and a structured record (JSON/CSV) is appended to the user's local history log.

### b. Description of Components
This section details the functional specifications of the hardware and software modules comprising the CalEyeZ architecture.

**A. Hardware Components**
1.  **Swan Digital Scale (Load Sensor):**
    * **Function:** Converts physical force into digital signals.
    * **Spec:** Equipped with a precision Strain Gauge Load Cell.
    * **Interface:** Transmits data wirelessly via a BLE (Bluetooth Low Energy) module using a custom GATT Serial profile.
    * **Output:** Hexadecimal packets containing weight values and status flags.
2.  **Optical Sensor (Camera):**
    * **Function:** Captures high-resolution visual data of the food items.
    * **Spec:** RGB Web Camera (1080p resolution).
    * **Interface:** Wired USB 2.0/3.0 connection for low-latency frame streaming.
    * **Role:** Provides the input matrix for the Computer Vision algorithm.

**B. Software Modules (Edge Application)**
3.  **BLE Communication Handler:**
    * **Role:** Manages the wireless link with the scale.
    * **Key Activities:** Scans for the device, handles connection handshakes, subscribes to "Notify" characteristics, and parses incoming HEX payloads. It also implements Checksum Validation to ensure data integrity and filter out corrupted packets.
4.  **AI Inference Engine:**
    * **Role:** Performs the visual recognition task.
    * **Technology:** YOLOv8 (You Only Look Once) deep neural network.
    * **Operation:** Runs locally on the device (Edge Computing). It accepts an image matrix as input and outputs a list of detected objects, including Class Labels (e.g., "Apple") and Confidence Scores (0-100%).
5.  **Fusion Logic Controller:**
    * **Role:** The central "brain" of the application.
    * **Key Activities:** Synchronizes the asynchronous threads (BLE stream vs. UI events), performs the mathematical calculation of calories/nutrients based on weight and type, and manages error handling (e.g., "Object not identified").
6.  **GUI/Dashboard:**
    * **Role:** The Human-Machine Interface (HMI).
    * **Technology:** Python Tkinter / CustomTkinter.
    * **Features:** Displays the live camera feed, real-time weight graph, nutritional breakdown cards, and control buttons.
7.  **Nutritional Database Module:**
    * **Role:** Provides the raw data for calculation.
    * **Operation:** Acts as a lookup table (Local DB or cached API response) mapping food labels (e.g., "Banana") to their macronutrient profiles (Calories, Protein, Fat, Carbs per 100g).

### c. GUI
The CalEyeZ user interface is designed with a focus on usability and real-time data visualization. Built using the CustomTkinter framework, it provides a modern, high-contrast dashboard that allows users to monitor the fusion process between the physical weight and the visual recognition.

**Key Interface Elements:**
1.  **Live Monitoring Panel:**
    * **Video Feed:** A central window displaying the raw RGB stream from the camera, allowing the user to ensure the food item is centered and visible.
    * **Real-Time Weight Graph:** A dynamic line chart that plots the weight signal (g) over time. This visual feedback is critical for the user to verify signal stability before triggering the analysis.
2.  **Control Cluster:**
    * **"Analyze" Trigger:** A prominent button that initiates the capture and inference sequence.
    * **Status Indicators:** Visual labels indicating the connection status of the BLE Scale (Connected/Scanning) and the Camera.
3.  **Nutritional Results Cards:**
    * Upon successful inference, the system generates distinct cards displaying the calculated values: Calories, Protein, Carbs, and Fat.
    * The recognized food label (e.g., "Food: Green Apple") is displayed prominently with the confidence score.
4.  **History & Analytics Dashboard:**
    * A secondary view providing tabular logs of past meals (CSV content) and graphical summaries (Bar Charts) of daily intake vs. goals.

### d. BOM-Bill of Materials

| # | Component Name | Specification/Description | Qty | Acquisition Status |
| :--- | :--- | :--- | :--- | :--- |
| 1 | Digital Smart Scale | BLE 4.0+ Support, High Precision Load Cell (Sensors), Custom GATT Profile. | 1 | Purchased |
| 2 | Generic Web Camera | 1080p Resolution @30fps. Standard RGB Sensor via USB interface. | 1 | Existing Inventory |
| 3 | Desktop Workstation | Stationary PC with NVIDIA GPU (CUDA Support) for YOLOv8 Training & Inference. | 1 | Existing Inventory |
| 4 | Calibration Objects | Known mass reference objects (e.g., sealed consumer goods) for sensor verification. | Var | Ad-Hoc/Available |
| 5 | Software Environment | Python 3.10, PyTorch, OpenCV, Bleak (BLE Library), CustomTkinter. | | Open Source |

## 6. Risk Management

| Risk Description | Likelihood | Severity | Mitigation Strategy |
| :--- | :--- | :--- | :--- |
| **BLE Connection Instability** | Medium | High | Implemented an auto-reconnect watchdog in the Python driver. The system buffers the last known stable weight if a packet drop occurs for less than 2 seconds. |
| **Poor Lighting Conditions** | High | Medium | The YOLOv8 model was trained with Data Augmentation (random brightness/contrast adjustments) to be robust. The GUI also alerts the user if the frame is too dark. |
| **Misclassification of Similar Foods** | Medium | Medium | Implemented a User-Verification Loop. The system displays the detected item for confirmation. A "Manual Entry" button is integrated into the GUI, allowing the user to override the AI and input the food name directly if detection fails. |
| **Missing USDA Nutritional Data** | Medium | Low | Implemented a Heuristic Filter ("Smart Search") that cleans query strings (e.g., removing "Raw" or "Frozen") to find the closest match in the database. |
| **USDA API Unavailability (Offline)** | Medium | High | Implemented Offline Redundancy. The system maintains a local JSON/SQL Database of common foods. If the external API is unreachable, the system automatically retrieves values from the local storage to ensure continuous operation. |

## 7. Verification
*(Detailed verification processes map into System Integration and V&V listed in Project Management).*

## 8. Project Management

### a. Schedule
*(Project Schedule: Pivot to BLE Implementation)*
1.  **Literature & Setup** (Completed: Oct '25)
2.  **Data Collection (16k)** (Completed: Oct '25 - Nov '25)
3.  **YOLOv8 Training** (Completed: Nov '25 - Dec '25)
4.  **OCR Feasibility (Pivoted)** (Pivoted/Dropped: Dec '25 - Jan '26)
5.  **GUI Development (Tkinter)** (Completed: Jan '26 - Feb '26)
6.  **BLE Reverse Eng.** (Active Phase: Feb '26)
7.  **System Integration** (Future Plan)
8.  **V&V & Testing** (Future Plan) -> **Final Submission**

### b. Team Roles

**Raz Dvora:**
* **Algorithm Lead:** Designed the Ensemble Architecture & XGBoost Router to dynamically switch between global and local models based on feature extraction.
* **Deep Learning:** Trained & optimized YOLOv8 custom models, fine-tuning hyperparameters for maximum precision on the Israeli dataset.
* **Data Strategy:** Managed the end-to-end data pipeline, including dataset collection, annotation, and class balancing via augmentation techniques.
* **System Logic:** Developed the core Python controller, BLE drivers, and the main decision logic (Fusion Engine).

**Roi Tzur:**
* **Frontend Lead (GUI):** Designed and implemented the user interface using CustomTkinter, focusing on real-time visualization of the camera feed and weight graphs.
* **CV Research (OCR):** Conducted the feasibility study for Optical Character Recognition, analyzing Tesseract/EasyOCR performance on digital screens (Outcome: Pivot to BLE).
* **Quality Assurance (QA):** Executed system integration tests (SIT) and validated the nutritional calculation accuracy against ground truth labels.
* **Usability Testing:** Managed user acceptance testing (UAT) to optimize the dashboard workflow and responsiveness.
