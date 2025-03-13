# Text-Human-Emotions-Detection-App

A web application for detecting six human emotions from text using machine learning. This project leverages NLP techniques and classic machine learning algorithms to classify input text into one of six emotions: **Joy, Fear, Anger, Love, Sadness,** and **Surprise**.

## Overview

This application was built with [Streamlit](https://streamlit.io/) and uses the following core components:

- **Text Processing:**
  - Cleans the text by removing non-alphabetic characters
  - Converts text to lowercase
  - Tokenizes and applies stopword removal and stemming (using NLTK)
- **Feature Extraction:**
  - Uses a TF-IDF vectorizer to transform input text into numerical features
- **Prediction:**
  - Uses a pre-trained Logistic Regression model to classify the text
  - A label encoder decodes the prediction back into an emotion

## Technical Stack

- **Python**
- **Streamlit** – for the web interface
- **scikit-learn** (v1.3.2) – for the Logistic Regression model and TF-IDF vectorizer
- **NumPy**
- **NLTK** – for natural language processing (stopwords, tokenization, stemming)
- **TensorFlow** (v2.15.0) *(if used in other parts of the project)*

## File Structure

```
app.py                        # Main Streamlit application file
logistic_regresion.pkl        # Pre-trained Logistic Regression model
tfidf_vectorizer.pkl          # Saved TF-IDF vectorizer
label_encoder.pkl             # Label encoder for decoding predicted labels
train.txt                     # Training data file (sample texts and labels)
vocab_info.pkl                # Additional vocabulary information (if used)
README.md                     # This file

Additional Files:
Six text Human Emotions Detection App.pptx  # Presentation overview
New Text Document.txt                       # Quick run command
Emotions Classification using ML and DL.ipynb (not included in repo)
```

## Installation

1. **Clone the repository:**

```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
```

2. **Set up a virtual environment (optional but recommended):**

```bash
python -m venv env
source env/bin/activate   # On Windows: env\Scripts\activate
```

3. **Install the required dependencies:**

```bash
pip install -r requirements.txt
```

4. **Download NLTK stopwords:**

```python
import nltk
nltk.download('stopwords')
```

## Running the App

To run the application locally, execute the following command:

```bash
python -m streamlit run app.py
```

This will start the Streamlit server and open the web app in your default browser.

## Customization and UI Enhancements

The UI was enhanced with advanced CSS for a modern look:

- **Animated background:** A dynamic gradient background that transitions smoothly.
- **Live glow effect on the title:** The main title features a pulsating glow effect.
- **Modern input styling:** Clean text area and buttons for a user-friendly interface.

## Training Data

The training data used for building the model is provided in `train.txt`. The data includes sample sentences with corresponding emotion labels, for example:

```
i didnt feel humiliated;sadness
i am feeling grouchy;anger
i feel romantic too;love
```

## Model Files

- **logistic_regresion.pkl:** Contains the trained Logistic Regression model.
- **tfidf_vectorizer.pkl:** Contains the saved TF-IDF vectorizer.
- **label_encoder.pkl:** Contains the label encoder used to decode numerical predictions to actual emotions.
- **vocab_info.pkl:** (if applicable) Contains additional vocabulary details used during training.

## Contribution

Feel free to fork the repository and submit pull requests for improvements. When contributing, please adhere to the following:

- Maintain code style consistency.
- Ensure that any UI modifications do not break the application.
- Update documentation as necessary.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
