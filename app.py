import os
from flask import Flask, request, redirect, url_for, render_template
import pandas as pd
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


app = Flask(__name__)

# Define the folder where uploaded files will be stored
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create the uploads folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Define the permanent CSV file name
PERMANENT_CSV_FILENAME = 'all_data.csv'
PERMANENT_CSV_PATH = os.path.join(app.config['UPLOAD_FOLDER'], PERMANENT_CSV_FILENAME)

# Initialize an empty DataFrame to store all uploaded data
if os.path.exists(PERMANENT_CSV_PATH):
    all_data_df = pd.read_csv(PERMANENT_CSV_PATH)
else:
    all_data_df = pd.DataFrame()

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    global all_data_df  # Declare as global here
    success = False  # Initialize success variable

    if request.method == 'POST':
        # Check if a file was submitted with the request
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        # Check if the file has a valid name and extension
        if file.filename == '':
            return redirect(request.url)

        if file:
            # Save the uploaded file to the uploads folder
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            # Read the uploaded file into a Pandas DataFrame
            with open(file_path, 'r') as txt_file:
                text_data = txt_file.read()

            # Create a new DataFrame with a "tags" column containing the text data
            df_new = pd.DataFrame({'tags': [text_data]})

            # Append the data to the DataFrame containing all uploaded data
            all_data_df = pd.concat([all_data_df, df_new], ignore_index=True)

            # Save the combined data as the permanent CSV file
            all_data_df.to_csv(PERMANENT_CSV_PATH, index=False)

            # Clean up: Delete the uploaded file after processing
            os.remove(file_path)

            success = True  # Set success to True after successful upload

    return render_template('upload.html', success=success)

@app.route("/cal", methods=["GET", "POST"])
def cal():
    ps = PorterStemmer()
    data_df = pd.read_csv('uploads/all_data.csv')

    def stem(text):
        y = []
        counter = 0
        for i in text.split():
            y.append(ps.stem(i))

        return " ".join(y)

    all_data_df['tags'] = all_data_df['tags'].apply(stem)

    cv = CountVectorizer(max_features=5000, stop_words='english')
    vector = cv.fit_transform(data_df['tags']).toarray()
    similarity = cosine_similarity(vector)
    # print(similarity)
    sim =round( similarity[0][1] * 100,2)

    
    return render_template('upload.html', similarity=sim)

if __name__ == '__main__':
    app.run(debug=True)