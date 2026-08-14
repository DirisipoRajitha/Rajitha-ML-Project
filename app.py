from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

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
    if food_name == "Tteokbokki":
        return """
<b>Health benefits for Indian people:</b>
- Provides quick energy from rice cakes.
- Contains some protein from fish cakes.
- Can be enjoyed as an occasional snack.

<b>Possible side effects for Indian people:</b>
- Very spicy and high in sodium.
- May trigger acidity or indigestion in some people.

<b>Best way for Indians to eat it:</b>
- Eat a small portion with vegetables or protein.
- Avoid eating it on an empty stomach.
- Drink water or buttermilk after eating if you are sensitive to spice.
"""

    elif food_name == "Kimchi":
        return """
<b>Health benefits for Indian people:</b>
- Rich in probiotics that support digestion and gut health.
- Contains vitamins A, C, and K.
- May improve immunity and support healthy digestion.

<b>Possible side effects for Indian people:</b>
- High sodium content may not be suitable for people with high blood pressure.
- Spicy fermented foods may cause acidity in some individuals.

<b>Best way for Indians to eat it:</b>
- Eat 1–2 tablespoons with rice, dal, roti, or khichdi.
- Pair it with paneer, egg, fish, or chicken.
"""

    elif food_name == "Bibimbap":
        return """
<b>Health benefits for Indian people:</b>
- Balanced meal with vegetables, rice, and protein.
- Good source of fiber and vitamins.

<b>Possible side effects for Indian people:</b>
- Gochujang can be spicy for some people.

<b>Best way for Indians to eat it:</b>
- Mix with vegetables and a protein source like egg or paneer.
"""

    elif food_name == "Bulgogi":
        return """
<b>Health benefits for Indian people:</b>
- High in protein and iron.
- Supports muscle health.

<b>Possible side effects for Indian people:</b>
- Can be high in sugar and sodium.

<b>Best way for Indians to eat it:</b>
- Pair with rice and vegetables.
"""

    elif food_name == "Japchae":
        return """
<b>Health benefits for Indian people:</b>
- Contains vegetables and some fiber.
- Light and easy to digest.

<b>Possible side effects for Indian people:</b>
- Noodles are high in carbohydrates.

<b>Best way for Indians to eat it:</b>
- Eat with vegetables and a protein source.
"""

    elif food_name == "Kimbap":
        return """
<b>Health benefits for Indian people:</b>
- Balanced combination of rice, vegetables, and protein.
- Good for a light meal.

<b>Possible side effects for Indian people:</b>
- Can be high in refined carbohydrates.

<b>Best way for Indians to eat it:</b>
- Eat fresh and pair with soup or salad.
"""

    return "Information not available."

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