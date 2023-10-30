from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from PIL import Image
from fastai.vision.all import *
from io import BytesIO
import openai

if not hasattr(Image, 'Resampling'):
    Image.Resampling = Image

app = FastAPI()

# openai.api_key = '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'

model = load_learner('densenet_wound_classifier.pkl')

def read_file_as_image(data) -> np.array:
    image = Image.open(BytesIO(data))
    image = np.array(image)
    return image

@app.get("/")
async def root():
    return {"message": "Welcome!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    predictions = model.predict(image)
    confidence_level = np.round(float(predictions[2].max() * 100), 2)
    return {
        'class': predictions[0],
        'confidence': confidence_level
    }

@app.post("/raphbot")
async def raphbot(message: dict):
    if message["msg"] == "quit":
        return {"Bot: Bye! Have a nice day!"}
    
    response = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages = [
            {"role": "assistant",
                "content": 
                """
                    You are Raph, a health assistant to help in providing information to users based on their health request. Make sure to be very polite
                    WHEN USERS ASK NON-HEALTH-RELATED QUESTIONS, tell them that you cannot help them with that and can only answer health related questions 💚, that they can ask any health relted question.
                    Be very CONVERSATIONAL and EMPATHETIC. Show that you care about their health and that's all that matters to you. Answers must SHORT, SIMPLE, CONCISE, and UNDERSTANDABLE. Use step-wise format for listing.
                    Ensure to get a clarified request before proceeding but do not be too pushy. Once you get some amount of information work with that and give the user a response.
                    During the conversation, if the user cut off from a particular complaints and moves to another different complaint, ensure to ask the user if they are done with the previous complaint, and if they say yes, then you can move on to the next complaint.
                    Identify if the user's case is severe or not, if severe, refer the user to a doctors that specialize on that case on the Raphina AI app. And if there is any issue of death or close to that, refer that user to the nearest hospital.
                    When ending the conversion, ask the user if there is anything else you can help with, and if there is nothing else, tell the user to have a nice day, and prioritize their health 💚.
                    When the user keeps saying bye and you are not sure if the user is done, ask the user if they are done, and if they say yes, then you can end the conversation and don't respond again.
                """
            }, 
            # Introduce yourself at the beginning of the conversation.
            {"role": "user", "content": message["msg"]}
        ]
    )
    bot_response = response['choices'][0]['message']['content']
    return (f"Bot: {bot_response}")

if __name__ == "__main__":
    uvicorn.run(app, host = "0.0.0.0", port = 8000)
