run these commands on terminal of VS Code after opening the folder having all the files.
python -m venv .venv

Activate the Environment:
.\.venv\Scripts\activate

Instal dependencies(wait for a 5-10 mins after running this):
pip install -r requirements.txt

update the .env file by putting your own Gemini API key.
i.e:
GOOGLE_API_KEY="YOUR_GEMINI_API_KEY_HERE"

Run the main.py
Start the AI microservice. It will run on http://127.0.0.1:5001:

Testing
The testAPI.py script runs a full sequence, including multi course indexing, grounding tests, and memory checks:
