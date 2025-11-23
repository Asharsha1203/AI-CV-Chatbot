import os
import json
import requests
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
import gradio as gr

# --- 1. Load Environment Variables ---
load_dotenv(override=True)
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
telegram_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"

# --- 2. Notification Functions (Modified from CarreerChatBot.ipynb, Cell 51) ---

def notify(message):
    """Sends a notification to Telegram and logs any error."""
    payload = {
        "chat_id": TELEGRAM_CHAT_ID, 
        "text": f"🤖 **CAREER CHATBOT LOG** 🤖\n\n{message}", 
        "parse_mode": "Markdown"
    }
    response = requests.post(telegram_url, data=payload)
    
    # Check for non-200 status codes and log the error
    if not response.ok:
        print(f"--- TELEGRAM NOTIFICATION ERROR ---")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        print(f"-----------------------------------")
    
    return response.status_code

# --- 3. Tool Functions (Modified from CarreerChatBot.ipynb, Cell 52) ---

def record_user_details(email, name="Name not provided", notes="not provided"):
    """Tool: Records user details and sends a Telegram notification."""
    notify(f"New visitor ➜ {name}, Email: {email}, Notes: {notes}")
    return {"recorded": "ok"}

def record_unknown_question(question):
    """Tool: Records an unanswered question and sends a Telegram notification."""
    notify(f"Unknown question logged ➜ {question}")
    return {"recorded": "ok"}

# --- 4. Tool Definitions (From CarreerChatBot.ipynb, Cells 53, 54, 55) ---
record_user_details_json = {
    "name": "record_user_details",
    "description": "Record when a user provides an email",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {"type": "string"},
            "name": {"type": "string"},
            "notes": {"type": "string"}
        },
        "required": ["email"]
    }
}

record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": "Record an unanswered question",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {"type": "string"}
        },
        "required": ["question"]
    }
}

tools = [
    {"type": "function", "function": record_user_details_json},
    {"type": "function", "function": record_unknown_question_json}
]

# --- 5. Main Chatbot Class ---
class CareerChatBot:
    
    def __init__(self):
        # Ollama Client setup (from CarreerChatBot.ipynb, Cell 50)
        self.client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama"  # dummy key
        )
        
        # Project-specific variables (from CarreerChatBot.ipynb, Cell 57)
        self.name = "A.S Harsha"
        
        # Read PDF content (using your specific file path)
        try:
            reader = PdfReader("me/Annabatula Sai Harsha CV Simple (1).pdf")
            self.linkedin = "".join([page.extract_text() or "" for page in reader.pages])
        except FileNotFoundError:
            self.linkedin = "Error: LinkedIn PDF file not found. Check file path."
            print("Error: Could not find me/Annabatula Sai Harsha CV Simple (1).pdf")
            
        # Read Summary content (using your specific file path)
        try:
            with open("me/new_summary.txt", "r", encoding="utf-8") as f:
                self.summary = f.read()
        except FileNotFoundError:
            self.summary = "Error: Summary file not found. Check file path."
            print("Error: Could not find me/new_summary.txt")

    def handle_tool_calls(self, tool_calls):
        """Processes tool calls, integrating notebook logic (Cell 56)."""
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            print(f"Tool called: {tool_name}", flush=True) 
            
            args = json.loads(tool_call.function.arguments)
            tool_fn = globals().get(tool_name)
            
            # Call the globally defined tool function
            result = tool_fn(**args)
            
            results.append({
                "role": "tool",
                "content": json.dumps(result),
                "tool_call_id": tool_call.id
            })
        return results

    def system_prompt(self):
        """Constructs the system prompt based on notebook logic (Cell 58)."""
        system_prompt = f"""
You are acting as {self.name}. Answer questions about {self.name}'s career, education, experience, projects, 
and background using the summary and LinkedIn profile below.

If you cannot answer something → call record_unknown_question.
If the user shows interest → ask for email and call record_user_details.

Be professional, friendly, and helpful.

## Summary:
{self.summary}

## LinkedIn:
{self.linkedin}

Stay in character as {self.name} throughout.
"""
        return system_prompt

    def chat(self, message, history):
        """The main chat loop (from CarreerChatBot.ipynb, Cell 59)."""
        messages = [{"role": "system", "content": self.system_prompt()}] \
                    + history \
                    + [{"role": "user", "content": message}]

        done = False
        while not done:
            response = self.client.chat.completions.create(
                model="llama3.2",     # Your Llama model from Cell 59
                messages=messages,
                tools=tools
            )

            finish = response.choices[0].finish_reason

            if finish == "tool_calls":
                tool_calls = response.choices[0].message.tool_calls
                results = self.handle_tool_calls(tool_calls)
                messages.append(response.choices[0].message)
                messages.extend(results)
            else:
                done = True

        return response.choices[0].message.content

# --- 6. Main Execution Block ---
if __name__ == "__main__":
    # Test the notification function on startup
    notify(f"Bot starting up! Chat ID: {TELEGRAM_CHAT_ID}")
    
    bot = CareerChatBot()
    gr.ChatInterface(bot.chat, type="messages").launch(share=True)