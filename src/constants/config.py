import os
from dotenv import load_dotenv

load_dotenv(override=True)  # Load the .env file

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

# Debugging output (optional)
if OPENAI_API_KEY:
    print(f"OpenAI API Key exists and begins {OPENAI_API_KEY[:8]}")
else:
    print("OpenAI API Key not set")

if ANTHROPIC_API_KEY:
    print(f"Anthropic API Key exists and begins {ANTHROPIC_API_KEY[:7]}")
else:
    print("Anthropic API Key not set")

if GOOGLE_API_KEY:
    print(f"Google API Key exists and begins {GOOGLE_API_KEY[:8]}")
else:
    print("Google API Key not set")