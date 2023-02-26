from flask import Flask, request, redirect
from twilio.twiml.messaging_response import MessagingResponse
from textblob import TextBlob



import csv

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
    choice = body.split(",", 1)[0].lower()
    print(choice)
    if choice == "about":
        resp.message("You have reached MedText, a SMS application for receiving AI input on your symptoms")
    elif choice == "symptoms":
        reader = csv.reader(body.split('\n'), delimiter=',')
        for row in reader:  # each row is a list
            symptomsList.append(row)
        for s in symptomsList[0][1:]:
            j = s.lower().stripls
            sentence = TextBlob(j)
            result = sentence.correct()
            symptoms.append(str(result))
        resp.message("Symptoms received, we will get back to you in a moment...\n")
    else:
        resp.message("Invalid input for the MedTex service. To perform a query, enter 'symptoms' follow by a comma "
                     "separated list of the symptoms you are experiencing. \n Ex: symptoms, headache,fever")

    print(symptoms)
    return str(resp)


if __name__ == "__main__":
    app.run(debug=True)
