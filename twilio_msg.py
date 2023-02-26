"""
file: twilio_msg.py
description: Driver code for twilio messaging service & AI connections
language: python3
author: Andrew Bush (apb2471@rit.edu)
"""

from flask import Flask, request, redirect
from twilio.twiml.messaging_response import MessagingResponse
from textblob import TextBlob

import csv
import openai
openai.api_key = "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"


def generate_response(prompt):
    model_engine = "text-davinci-003"
    prompt = (f"{prompt}")

    completions = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.85,
    )

    message = completions.choices[0].text
    return message.strip()


app = Flask(__name__)


@app.route("/sms", methods=['GET', 'POST'])
def incoming_sms():
    """Send a dynamic reply to an incoming text message"""
    # Get the message the user sent our Twilio number
    body = request.values.get('Body', None)
    # Start our TwiML response
    resp = MessagingResponse()

    # Determine the right reply for this message
    symptomsList = []
    symptoms = []
    choice = body.split(" ", 1)[0].lower()
    if choice == "about":
        resp.message("You have reached MedText, a SMS-based application for providing AI input on your medical symptoms")
    elif choice == "symptoms":
        reader = csv.reader(body.split('\n'), delimiter=',')
        for row in reader:  # each row is a list
            symptomsList.append(row)
        for s in symptomsList[0][1:]:
            j = s.lower().strip()
            sentence = TextBlob(j)
            result = sentence.correct()
            symptoms.append(str(result))

        prompt = "What illness may is associated with the following symptoms: " + str(symptoms)
        response = generate_response(prompt)
        resp.message("AI Analysis: " + response)
    else:
        resp.message("Invalid input for the MedTex service. To perform a query, enter 'symptoms' follow by a comma "
                     "separated list of the symptoms you are experiencing. \n Ex: symptoms headache, fever, sore throat")

    print(symptoms)
    return str(resp)


if __name__ == "__main__":
    app.run(debug=True)
