# AI-Powered Online Resume Screening System

![Resume Screening Banner](https://user-images.githubusercontent.com/79269520/150174169-1ddc4349-3e3a-4395-8167-154a43b952a2.png)

## 📖 Overview

This project is a sophisticated, end-to-end **AI-Powered Resume Screening System** designed to automate and enhance the initial stages of the recruitment process. The system leverages state-of-the-art Natural Language Processing (NLP) to intelligently rank candidates by semantically matching their resumes against a given job description.

The core of the project is a powerful web application built with **Streamlit**. Recruiters can upload a candidate's resume (in PDF, DOCX, or TXT format), and the application will instantly provide a ranked list of the most suitable job openings from its database. Each match includes a similarity score and a list of common skills found in both the resume and the job description, providing transparent and actionable insights.

This tool transforms a time-consuming manual task into a fast, data-driven, and efficient process.

---

## 📂 Repository Contents

This repository is structured to showcase both the final application and the development process behind it.

1.  **`app.py`**: The main Python script that runs the interactive Streamlit web application. This is the final, deployable product.
2.  **`resume_screening.ipynb`**: A comprehensive Jupyter Notebook that documents the entire development journey. It covers data loading, text preprocessing, the generation of semantic embeddings using Sentence Transformers, and the logic for similarity matching.
3.  **`resume_embeddings.npy`**: A NumPy file containing the pre-computed sentence embeddings for all resumes in the dataset. Storing these saves significant processing time when the app is running.
4.  **`job_embeddings.npy`**: A NumPy file with the pre-computed embeddings for all job descriptions.

*(Note: The datasets `Resume.csv` and `job_title_des.csv` are not uploaded to this repository due to their size. Please see the Dataset section for instructions.)*

---

## 💾 Dataset

This project relies on two distinct datasets to function:

1.  **Resume Dataset:** A collection of resumes from various candidates. This project uses a dataset commonly found on Kaggle containing thousands of resumes across different job categories.
    *   **Recommended Link:** [Resume Dataset on Kaggle](https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset)
2.  **Job Description Dataset:** A collection of job postings with detailed descriptions.
    *   **Recommended Link:** [Job Description Dataset on Kaggle](https://www.kaggle.com/datasets/kshitizregmi/jobs-and-job-description)

**Instructions:**
To run this project, you must download both datasets. After unzipping, ensure the folder structure in your local project directory matches the following:
```
your-project-folder/
├── Resume.csv/
│   └── Resume.csv
├── job_title_des.csv/
│   └── job_title_des.csv
├── app.py
└── resume_screening.ipynb
... (and other files)
```

---

## ✨ Core Features & Project Workflow

1.  **Data Loading & Preprocessing:** Resumes and job descriptions are loaded from CSV files using Pandas.
2.  **Semantic Embedding Generation:** The powerful **Sentence-Transformers** library (`all-MiniLM-L6-v2` model) is used to convert the text of each resume and job description into high-dimensional numerical vectors (embeddings). These embeddings capture the semantic meaning of the text, going far beyond simple keyword matching.
3.  **Real-time Matching:** When a user uploads a resume, its text is extracted and converted into an embedding. **Cosine Similarity** is then used to calculate the semantic similarity between the uploaded resume and all job descriptions in the database.
4.  **Skill Extraction:** The **spaCy** library is used to perform Named Entity Recognition (NER) by matching against a custom list of technical skills. The application identifies and displays the skills that are common to both the resume and the top-matching job descriptions.
5.  **Interactive Web Application:** A user-friendly front-end built with **Streamlit** allows users to easily upload files (PDF, DOCX, TXT) and view the ranked results in an intuitive, expandable format.

---

## 🛠️ Tech Stack & Libraries

*   **Language:** Python
*   **Web Framework:** Streamlit
*   **Core NLP/ML:**
    *   **Sentence-Transformers:** For generating semantic text embeddings.
    *   **Scikit-learn:** For calculating Cosine Similarity.
    *   **spaCy:** For NER-based skill extraction.
    *   **NLTK:** For text preprocessing tasks.
*   **Data Handling:** Pandas, NumPy
*   **File Parsing:** PyPDF2, python-docx

---

## 🚀 How to Run the Application

To run the interactive web application on your local machine:

1.  **Clone the repository:**
    ```
    git clone https://github.com/your-username/your-repo-name.git
    ```
2.  **Navigate to the project directory:**
    ```
    cd your-repo-name
    ```
3.  **Install the required libraries:**
    *(It is highly recommended to use a virtual environment)*
    ```
    pip install -r requirements.txt
    ```
    *(Note: You will need to create this `requirements.txt` file first by running `pip freeze > requirements.txt` in your terminal.)*
4.  **Download spaCy Model:**
    ```
    python -m spacy download en_core_web_sm
    ```
5.  **Download and place the datasets** as described in the "Dataset" section above.
6.  **Generate Embeddings (First-time setup):**
    Open the `resume_screening.ipynb` notebook and run the cells related to data loading and embedding generation to create the `resume_embeddings.npy` and `job_embeddings.npy` files.
7.  **Launch the Streamlit App:**
    ```
    streamlit run app.py
    ```
    Your web browser will automatically open with the application running.

---


