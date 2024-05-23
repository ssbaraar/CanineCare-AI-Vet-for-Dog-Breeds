import json
import os
from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import sklearn
import joblib  # For loading the trained model

app = Flask(__name__)

# Load the machine learning model for breed classification
path = 'model/20220804-16551659632113-all-images-Adam.h5'
custom_objects = {'KerasLayer': hub.KerasLayer}
with tf.keras.utils.custom_object_scope(custom_objects):
    model = tf.keras.models.load_model(path)

# Load the model (ensure the path to your model file is correct)
# TensorFlow model
keras_model = tf.keras.models.load_model(path, custom_objects=custom_objects)

# Scikit-learn model loaded via joblib
joblib_model = joblib.load('model/dogModel1.pkl')


# Load the data and preprocess it
df = pd.read_csv("data/dog_data_09032022.csv")
# Perform data cleaning and preprocessing steps from the provided code
# 5)
def is_number(string):
    try:
        float(string)
        return True
    except ValueError:
        return False

def hwl_clean(height): # height weight life clean
    height = str(height)
    height_list = height.split('-')
    result = []
    for word in height_list:
        result = result + word.split(" ")
    avg_val = 0
    count = 0
    for i in result:
        if is_number(i):
            count = count +1
            avg_val = avg_val + float(i)
    if count != 0:
        avg_val = avg_val / count
        return avg_val
    else:
        return 0

df['height_c'] = df['height'].apply(hwl_clean)
df['weight_c'] = df['weight'].apply(hwl_clean)
df['life_c'] = df['life'].apply(hwl_clean)

# replace the outlier with the mean value
mean_val = (df["life_c"].sum() - df["life_c"].max()) / 275
df["life_c"].replace(df["life_c"].max(), mean_val, inplace= True)

df.drop(['height','weight','life'], axis = 1,inplace = True)

#6)
df[['height_c','weight_c','life_c']].isnull().sum()

#7)
def popular_clean(pop):
    if type(pop) == float:
        return np.nan
    else:
        rank_list = pop.split(" of ")
        measure = int(rank_list[0]) / int(rank_list[1])
        return 1 - measure

df['popularity_rank_c']= df['popularity_rank'].apply(popular_clean)
df.drop('popularity_rank', axis = 1, inplace = True)
df['Coat Length'].value_counts()

#8)
def coatLen_clean(coat):
    if type(coat) != float:
        if "Long" in coat:
            return 3
        elif "Medium" in coat:
            return 2
        else:
            return 1
    else:
        return np.nan

df["Coat_Length_c"] = df["Coat Length"].apply(coatLen_clean)
df.drop("Coat Length", axis = 1, inplace= True)

#9)
coat_dict = {}
for i in df["Coat Type"]:
    if type(i) != float:
        coat_list = i.split('-')
        for coat in coat_list:
            if coat not in coat_dict:
                coat_dict[coat] = 1
            else:
                coat_dict[coat] = coat_dict[coat] + 1

new_dict = dict(sorted(coat_dict.items(), key=lambda item: item[1], reverse= True))
new_dict

#10)
def coat_clean(coat):
    if type(coat) != float:
        if 'Double' in coat:
            return 'Double'
        elif 'Smooth' in coat:
            return 'Smooth'
        elif 'Wiry' in coat:
            return 'Wiry'
        elif 'Silky' in coat:
            return 'Silky'
        elif 'Curly' in coat:
            return 'Curly'
        elif 'Rough' in coat:
            return 'Rough'
        elif 'Corded' in coat:
            return 'Corded'
        elif 'Hairless' in coat:
            return 'Hairless'
    else:
        return np.nan
df['coat_c'] = df['Coat Type'].apply(coat_clean)
df.drop("Coat Type", axis = 1, inplace = True)
df.drop("marking", axis = 1, inplace = True)
df.drop("color", axis = 1, inplace = True)
# Normalized Euclidean columns
euclidean_cols = ['height_c', 'weight_c', 'life_c',
                  'Coat_Length_c', 'Affectionate With Family',
                  'Good With Young Children', 'Good With Other Dogs', 'Shedding Level',
                  'Coat Grooming Frequency', 'Drooling Level', 'Openness To Strangers',
                  'Playfulness Level', 'Watchdog/Protective Nature', 'Adaptability Level',
                  'Trainability Level', 'Energy Level', 'Barking Level',
                  'Mental Stimulation Needs']
df_euclidean = df[euclidean_cols]
df_euclidean.fillna(df_euclidean.mean(), inplace=True)
normalized_df_euclidean = (df_euclidean - df_euclidean.min()) / (df_euclidean.max() - df_euclidean.min())

# Function to get the most similar breeds
def get_names(profile):
    dist_list = []
    for i in range(normalized_df_euclidean.shape[0]):
        dist_list.append(np.linalg.norm(profile - normalized_df_euclidean.iloc[i]))
    idx_list = sorted(range(len(dist_list)), key=lambda i: dist_list[i], reverse=False)[:5]
    names = df['dog'][idx_list].values
    distances = 1 - (sorted(dist_list)[0:5] / sorted(dist_list)[-1])
    return names, distances

@app.route('/recommend_breed', methods=['GET', 'POST'])
def recommend_breed():
    if request.method == 'POST':
        json_data = request.data.decode('utf-8')

        # Parse the JSON data into a Python dictionary
        data = json.loads(json_data)

        # Access each JSON object
        for key, value in data.items():
            print(key, value)
        # Get the form data
        height = float(data['height_c'])
        print("Height", height)
        weight = float(data['weight_c'])
        life = float(data['life_c'])
        coat_length = float(data['coat_length_c'])
        affection = float(data['affection_c'])
        good_with_kids = float(data['good_with_kids_c'])
        good_with_dogs = float(data['good_with_dogs_c'])
        shedding = float(data['shedding_c'])
        grooming = float(data['grooming_c'])
        drooling = float(data['drooling_c'])
        openness = float(data['openness_c'])
        playfulness = float(data['playfulness_c'])
        watchdog = float(data['watchdog_c'])
        adaptability = float(data['adaptability_c'])
        trainability = float(data['trainability_c'])
        energy = float(data['energy_c'])
        barking = float(data['barking_c'])
        mental_stimulation = float(data['mental_stimulation_c'])

        # Normalize the input values
        profile = [(height - 6.5) / 32.0, (weight - 5) / 195, (life - 6.5) / 18,
                   (coat_length - 1) / 3, (affection - 1) / 5, (good_with_kids - 1) / 5,
                   (good_with_dogs - 1) / 5, (shedding - 1) / 5, (grooming - 1) / 5,
                   (drooling - 1) / 5, (openness - 1) / 5, (playfulness - 1) / 5,
                   (watchdog - 1) / 5, (adaptability - 1) / 5, (trainability - 1) / 5,
                   (energy - 1) / 5, (barking - 1) / 5, (mental_stimulation - 1) / 5]

        # Get the recommended breeds
        names, distances = get_names(profile)

        # Convert the data to a list of dictionaries
        data = [{'name': name, 'similarity': distance} for name, distance in zip(names, distances)]

        return jsonify(data)



############################################DONT TOUCH THIS##############################################
# Define the list of labels
labels = ['affenpinscher', 'afghan_hound', 'african_hunting_dog', 'airedale', 'american_staffordshire_terrier', 'appenzeller', 'australian_terrier', 'basenji', 'basset', 'beagle', 'bedlington_terrier', 'bernese_mountain_dog', 'black-and-tan_coonhound', 'blenheim_spaniel', 'bloodhound', 'bluetick', 'border_collie', 'border_terrier', 'borzoi', 'boston_bull', 'bouvier_des_flandres', 'boxer', 'brabancon_griffon', 'briard', 'brittany_spaniel', 'bull_mastiff', 'cairn', 'cardigan', 'chesapeake_bay_retriever', 'chihuahua', 'chow', 'clumber', 'cocker_spaniel', 'collie', 'curly-coated_retriever', 'dandie_dinmont', 'dhole', 'dingo', 'doberman', 'english_foxhound', 'english_setter', 'english_springer', 'entlebucher', 'eskimo_dog', 'flat-coated_retriever', 'french_bulldog', 'german_shepherd', 'german_short-haired_pointer', 'giant_schnauzer', 'golden_retriever', 'gordon_setter', 'great_dane', 'great_pyrenees', 'greater_swiss_mountain_dog', 'groenendael', 'ibizan_hound', 'irish_setter', 'irish_terrier', 'irish_water_spaniel', 'irish_wolfhound', 'italian_greyhound', 'japanese_spaniel', 'keeshond', 'kelpie', 'kerry_blue_terrier', 'komondor', 'kuvasz', 'labrador_retriever', 'lakeland_terrier', 'leonberg', 'lhasa', 'malamute', 'malinois', 'maltese_dog', 'mexican_hairless', 'miniature_pinscher', 'miniature_poodle', 'miniature_schnauzer', 'newfoundland', 'norfolk_terrier', 'norwegian_elkhound', 'norwich_terrier', 'old_english_sheepdog', 'otterhound', 'papillon', 'pekinese', 'pembroke', 'pomeranian', 'pug', 'redbone', 'rhodesian_ridgeback', 'rottweiler', 'saint_bernard', 'saluki', 'samoyed', 'schipperke', 'scotch_terrier', 'scottish_deerhound', 'sealyham_terrier', 'shetland_sheepdog', 'shih-tzu', 'siberian_husky', 'silky_terrier', 'soft-coated_wheaten_terrier', 'staffordshire_bullterrier', 'standard_poodle', 'standard_schnauzer', 'sussex_spaniel', 'tibetan_mastiff', 'tibetan_terrier', 'toy_poodle', 'toy_terrier', 'vizsla', 'walker_hound', 'weimaraner', 'welsh_springer_spaniel', 'west_highland_white_terrier', 'whippet', 'wire-haired_fox_terrier', 'yorkshire_terrier']

# Function to preprocess the image and make a prediction
def predict_breed(image):
    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(image, axis=0)
    image = image.reshape((-1, 224, 224, 3))
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.constant(image)
    prediction = model.predict(image).flatten()
    predicted_breed = labels[np.argmax(prediction)]
    return predicted_breed

@app.route('/predict_breed_route', methods=['POST'])
def predict_breed_route():
    file = request.files['file']
    if file:
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        predicted_breed = predict_breed(image)
        return jsonify({'predicted_breed': predicted_breed})
    return jsonify({'error': 'No image provided'}), 400


@app.route('/', methods=['GET', 'POST'])
def index():
    predicted_breed = None

    if request.method == 'POST':
        file = request.files['file']
        if file:
            image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
            predicted_breed = predict_breed(image)

    return render_template('Home.html', predicted_breed=predicted_breed)


@app.route('/')
def home():
    symptoms = []
    return render_template('Home.html', symptoms=symptoms)

@app.route('/predict_disease', methods=['POST'])
def predict_disease():
    data = request.get_json()
    symptoms = data['symptoms']

    # Ensure the input vector is formatted correctly for the scikit-learn model
    input_vector = np.array(symptoms).reshape(1, -1)

    # Predict using the scikit-learn model loaded via joblib
    prediction = joblib_model.predict(input_vector)
    predicted_index = np.argmax(prediction)

    # Mapping of prediction index to disease name
    disease_mapping = {
        0: 'Tick fever', 1: 'Distemper', 2: 'Parvovirus',
        3: 'Hepatitis', 4: 'Tetanus', 5: 'Chronic kidney Disease',
        6: 'Diabetes', 7: 'Gastrointestinal Disease', 8: 'Allergies',
        9: 'Gingitivis', 10: 'Cancers', 11: 'Skin Rashes'
    }
    predicted_disease = disease_mapping.get(predicted_index, "Unknown Disease")

    # Return the prediction as a JSON response
    return jsonify(disease=predicted_disease)



if __name__ == '__main__':
    app.run(debug=True)
