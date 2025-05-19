import requests

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    response = requests.post(
        "http://127.0.0.1:5000/chat",
        json={"message": user_input}
    )
    print("Bot:", response.json().get("response"))
    