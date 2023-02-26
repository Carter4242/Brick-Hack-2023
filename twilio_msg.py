"""
file: twilio_msg.py
description: Driver code for twilio messaging service & AI connections
language: python3
author: Andrew Bush (apb2471@rit.edu)
"""

from flask import Flask, request, session
from twilio.twiml.messaging_response import MessagingResponse
from textblob import TextBlob

from SymptomSuggestion import twilioInputSymptoms, twilioMoreDetails, twilioPrintSymptoms

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


SECRET_KEY = 'a secret key'
app = Flask(__name__)
app.config.from_object(__name__)

callers = {
    "+13156728284 : Andrew"
}


@app.route("/sms", methods=['GET', 'POST'])
def incoming_sms():
    # Increment the counter
    counter = session.get('counter', 0)
    counter += 1
    session['counter'] = counter
    from_number = request.values.get('From')
    if from_number in callers:
        name = callers[from_number]
    else:
        name = "Friend"

    message = '{} has messaged {} {} times.' \
        .format(name, request.values.get('To'), counter)
    """Send a dynamic reply to an incoming text message"""
    # Get the message the user sent our Twilio number
    body = request.values.get('Body', None)
    # Start our TwiML response
    resp = MessagingResponse()
    choice = body.split(" ", 1)[0].lower()
    if choice == "reset":
        resp.message("Session reset.")
        session['counter'] = 0
        return str(resp)
    elif choice == "about":
        resp.message(
            "You have reached MedText, a SMS-based application "
            "for providing AI input on your medical symptoms")
        return str(resp)
    elif choice == "session":
        resp.message(message)
        return str(resp)
    if counter == 1:
        symptomsList = []
        symptoms = []
        reader = csv.reader(body.split('\n'), delimiter=',')
        for row in reader:  # each row is a list
            symptomsList.append(row)
        for s in symptomsList[0][1:]:
            j = s.lower().strip()
            sentence = TextBlob(j)
            result = sentence.correct()
            symptoms.append(str(result))
        response = twilioInputSymptoms(symptoms)
        resp.message(response)
    if counter == 2:
        response = str(twilioPrintSymptoms())
        resp.message(response)
    return str(resp)


if __name__ == "__main__":
    app.run(debug=True)
