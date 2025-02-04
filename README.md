### AGE_EMOTION_DETECTION_MOVIE_THEATRE

#### Age detection dataset : [https://drive.google.com/file/d/1KRzZ0TMhxLQru0WJGqmFLyGBor6mhkzW/view?usp=sharing]

#### Emotion detection dataset: [https://drive.google.com/file/d/1N67ZjB_T_7rzYqqBKo8ENSvdnh7Pc_xB/view?usp=sharing]

- Dataset Folder: Download and Extract the zip files and place them in the `Dataset` folder 

#### Models : [https://drive.google.com/file/d/1Dr15FeH617qlgGc72x0F_oLSnSiFKr7f/view?usp=sharing] 

- Download and Extract the Models and plce them in the project directory

Folder Structure:
```plaintext
AGE_EMOTION_DETECTION_MOVIE_THEATRE
│── CSV_File/
│   ├── detections.csv                 # Stores detected age & emotion data
│
│── Dataset/
│   ├── Emotions/                       # Emotion dataset (FER2013)
│   │   ├── test/
│   │   ├── train/
│   ├── UTKFace/                        # Age dataset (UTKFace)
│
│── Models/                             # Pre-trained deep learning models
│   ├── ageModel.h5
│   ├── ageModel2.h5
│   ├── emotionModel1.h5
│   ├── emotionModel2.h5
│   ├── emotionModel3.h5
│
│── Notebooks/                          # Jupyter notebooks for model training
│   ├── Age_Detection_model.ipynb
│   ├── Emotion_Detection_model.ipynb
│
│── Report/                             # Project documentation and analysis
│
│── app.py                              # Main application for real-time detection
│── requirements.txt                     # Required Python packages
```

How to Run:

- Step 1:
```bash
conda create -n theatre python=3.10 -y
```

- Step 2:
```bash
conda activate theatre
```

- Step 3:
```bash
pip install -r requirements.txt
```
- Step 4:
```bash
python app.py
```
