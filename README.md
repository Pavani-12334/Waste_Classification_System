Waste Classification System using CNN

This project uses Deep Learning (CNN) to automatically classify waste into different categories such as cardboard, glass, metal, paper, plastic, and trash.
It promotes sustainability by encouraging smart waste management and recycling.

📂 Dataset

The dataset used is from Kaggle:
🔗 Garbage Classification Dataset on Kaggle
https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification

Folder structure used in this project:

dataset/
 └── Garbage classification/
      ├── cardboard/
      ├── glass/
      ├── metal/
      ├── paper/
      ├── plastic/
      └── trash/

🧠 Model Overview

Algorithm: Convolutional Neural Network (CNN)

Framework: TensorFlow / Keras

Image size: 128x128 pixels

Optimizer: Adam

Loss Function: Categorical Crossentropy

Accuracy Goal: ~90% (depends on dataset quality)

⚙️ How to Run

Clone the repository:

git clone https://github.com/Pavani-12334/Waste_Classification_System.git
cd Waste_Classification_System


Install dependencies:

pip install tensorflow keras matplotlib numpy pillow


Place the dataset inside:

dataset/Garbage classification/


Run the training script:

python waste_classifier.py

🌍 Sustainability Impact

Reduces manual waste segregation.

Encourages recycling and eco-friendly waste management.

Can be extended with IoT-based smart bins to detect and sort waste automatically.

🧾 Results

CNN model trained successfully on multiple waste types.

Achieved accurate classification of recyclable vs non-recyclable items.

👩‍💻 Author

Pavani Padigela
🎓 B.Tech – Computer Science Engineering
💡 Passionate about AI, Deep Learning, and Sustainable Technology
