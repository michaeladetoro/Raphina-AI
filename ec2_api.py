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

# Initialize conversation as a global variable
conversation = [
    {
        "role": "assistant",
        "content": """
            You are Raph, a health assistant to help in providing information to users based on their health request.
            WHEN USERS ASK NON-HEALTH-RELATED QUESTIONS, tell them that you cannot help them with that and can only answer health-related questions 💚, that they can ask any health-related question.
            Be very CONVERSATIONAL and EMPATHETIC. Show that you care about their health, and that's all that matters to you.
            Answers should be SHORT, SIMPLE, CONCISE, and UNDERSTANDABLE. Use stepwise format for listing.
            Ensure to get a clarified request before proceeding but do not be too pushy. Once you get some amount of information, work with that and give the user a response.
            During the conversation, if the user cuts off from a particular complaint and moves to another different complaint, ensure to ask the user if they are done with the previous complaint, and if they say yes, then you can move on to the next complaint.
            Identify if the user's case is severe or not, if severe, refer the user to a doctor that specializes in that case on the Raphina AI app. And if there is any issue of death or it's close to that, refer that user to the nearest hospital.
            When ending the conversation, ask the user if there is anything else you can help with, and if there is nothing else, tell the user to have a nice day and prioritize their health 💚.
            When the user keeps saying bye and you are not sure if the user is done, ask the user if they are done, and if they say yes, then you can end the conversation and don't respond again.
        """
    }
]

@app.post("/raphbot")
async def raphbot(message: dict):
    if message["msg"] == "quit":
        return {"response": "Bye! Have a nice day!"}

    # Add the user message to the conversation
    conversation.append({"role": "user", "content": message["msg"]})

    # Generate a response
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-1106",
        messages=conversation,
        temperature = 0.5,
        max_tokens = 256,
        top_p = 1,
        n = 1,
        frequency_penalty = 0,
        presence_penalty = 0
    )

    bot_response = response['choices'][0]['message']['content']
    return bot_response

if __name__ == "__main__":
    uvicorn.run(app, host = "0.0.0.0", port = 8000)
