from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import re 
from google import genai
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Load trained model
model = load_model("korean_food_model_improved.h5")

# Class labels
class_names = ["Bibimbap", "Bulgogi", "Japchae", "Kimbap", "Kimchi", "Tteokbokki"]

# Food details
food_info = {
    "Bibimbap": {
        "ingredients": "Rice, mixed vegetables, egg, gochujang (Korean chili paste), sesame oil",
        "description": "A traditional Korean rice bowl mixed with vegetables, egg, and spicy chili paste."
    },
    "Bulgogi": {
        "ingredients": "Beef, soy sauce, garlic, sugar, sesame oil, onion",
        "description": "A famous Korean dish made with sweet and savory marinated beef."
    },
    "Japchae": {
        "ingredients": "Sweet potato noodles, vegetables, soy sauce, sesame oil",
        "description": "Stir-fried Korean glass noodles with vegetables and sesame flavor."
    },
    "Kimbap": {
        "ingredients": "Rice, seaweed, vegetables, egg, pickled radish",
        "description": "A Korean seaweed rice roll filled with vegetables and other ingredients."
    },
    "Kimchi": {
        "ingredients": "Napa cabbage, Korean chili powder, garlic, ginger, salt",
        "description": "A traditional fermented Korean cabbage dish rich in flavor and probiotics."
    },
    "Tteokbokki": {
        "ingredients": "Rice cakes, gochujang, fish cakes, onion, garlic",
        "description": "A popular spicy Korean street food made with chewy rice cakes."
    }
}
def generate_ai_info(food_name):
    prompt = f"""
You are a food and nutrition assistant.

The predicted Korean dish is: {food_name}

Generate information specifically for Indian users.

Give the answer in exactly these 3 sections:

### 1. Health Benefits

### 2. Possible Side Effects

### 3. Best Way to Eat for Indians

Keep the information simple, practical, and easy to understand.
Do not make medical claims.
Mention moderation where appropriate.

Dish: {food_name}
"""

    try:
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt
        )

        text = response.text

        text = re.sub(
            r'\*\*(.*?)\*\*',
            r'<strong>\1</strong>',
            text
        )

        text = re.sub(
            r'^###\s*(.*?)$',
            r'<h3>\1</h3>',
            text,
            flags=re.MULTILINE
        )

        text = re.sub(
            r'^---$',
            '<hr>',
            text,
            flags=re.MULTILINE
        )

        text = text.replace('\n', '<br>')

        return text

    except Exception as e:
        return f"Generative AI information could not be generated: {str(e)}"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file uploaded'

    file = request.files['file']

    if file.filename == '':
        return 'No file selected'

    filepath = os.path.join('static', file.filename)
    file.save(filepath)

    img = image.load_img(filepath, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = round(float(np.max(prediction)) * 100, 2)

    ai_text = generate_ai_info(predicted_class)

    return render_template(
        'index.html',
        prediction=predicted_class,
        confidence=confidence,
        image_file=file.filename,
        ingredients=food_info[predicted_class]['ingredients'],
        description=food_info[predicted_class]['description'],
        ai_text=ai_text
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
    app.run(host='0.0.0.0', port=5000, debug=True)
