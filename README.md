# 🧠 Nationality Detection Model with Conditional Predictions

This project is developed as part of an AI internship program. The goal is to build a machine learning system that predicts:

- 🌍 **Nationality** of a person from an image
- 😀 **Emotion** of the person
- 🧥 For Indian: also predict Age and Dress Color
- 🧓 For US: also predict Age
- 👗 For African: also predict Dress Color

## 📦 Project Structure

├── app.py # Streamlit GUI
├── preprocessor.py # Image preprocess logic
├── requirements.txt
├── models/ # Saved models (.h5 files)
├── gui/ # GUI elements & layout
├── train_nationality.ipynb
├── train_emotion.ipynb
├── train_age.ipynb
├── train_dresscolor.ipynb
└── dataset/
├── nationality/
├── emotion/
├── age/
└── dresscolor/



---

## 📥 Dataset Links

Download and extract into the `dataset/` folder:

- 🏳️ **Nationality Dataset**  
  https://www.kaggle.com/datasets/abhikjha/imdb-wiki-faces-dataset

- 😊 **Emotion Dataset**  
  https://www.kaggle.com/datasets/msambare/fer2013

- 👕 **Dress Color Dataset**  
  https://www.kaggle.com/datasets/grassknoted/dress-pattern-attribute-dataset  
  *(Use color labels only)*

- 🎂 **Age Dataset**  
  https://susanqq.github.io/UTKFace/  

---

## 🚀 Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/nationality-detection-task.git
cd nationality-detection-task

python -m venv venv
venv\\Scripts\\activate     # Windows
# or
source venv/bin/activate    # Linux/macOS

pip install -r requirements.txt


🛠️ Training the Models
Each task has its own training notebook:

Run in order:

train_nationality.ipynb

train_emotion.ipynb

train_age.ipynb

train_dresscolor.ipynb

Each will produce a .h5 file inside models/.



Run the Streamlit GUI
streamlit run app.py


--
Ritwika Saha