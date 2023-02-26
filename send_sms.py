# Download the helper library from https://www.twilio.com/docs/python/install
# MG36662aed6c2be3e2a4066d892c22b584

# Find your Account SID and Auth Token at twilio.com/console
# and set the environment variables. See http://twil.io/secure
# Download the helper library from https://www.twilio.com/docs/python/install
import os
from twilio.rest import Client


# Find your Account SID and Auth Token at twilio.com/console
# and set the environment variables. See http://twil.io/secure
account_sid = 'AC0a76125e48452ac66e8e8760d1213e31'
auth_token = '71a76c084d0e37af9164adcf82f4f44f'
client = Client(account_sid, auth_token)

message = client.messages \
    .create(
         messaging_service_sid='MG36662aed6c2be3e2a4066d892c22b584',
         body="""
            
         """,
         to='+15857546680'
     )

print(message.sid)
