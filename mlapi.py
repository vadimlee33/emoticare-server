import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = FastAPI()

maxlen = 50

# Define the input data model
class TextData(BaseModel):
    text: str

# Load the TensorFlow model
model = tf.keras.models.load_model('coursera_model_new.h5')

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Mapping for the label indices and their corresponding emotions
pred_dict = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}
index_to_class = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}

# API endpoint for text scoring
@app.post("/score_text")
def score_text(text_data: TextData):
    text = text_data.text
    test_sentences = []
    test_sentences.append(text)
    test_sequ = get_sequences(tokenizer, test_sentences)

    # Make predictions using the loaded TensorFlow model
    p = model.predict(np.expand_dims(test_sequ[0], axis=0))[0]
    pred_class = np.argmax(p).astype('uint8')
    predicted_emotion = pred_dict[pred_class]

    return {"text": text, "predicted_emotion": predicted_emotion}

def get_sequences(tokenizer, tweets):
    sequence = tokenizer.texts_to_sequences(tweets)
    padded = pad_sequences(sequence, truncating='post', padding='post', maxlen=maxlen)
    return padded


# Run the server with Uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
