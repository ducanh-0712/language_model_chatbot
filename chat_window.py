import torch
from ai_chatbot import GPTLanguageModel
from ai_chatbot import decode
from ai_chatbot import encode
from ai_chatbot import device
from ai_chatbot import checkpoint_file

checkpoint = torch.load(checkpoint_file, map_location=device)

max_new_tokens = 1000

model = GPTLanguageModel().to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

conversation = ""

while True:
    user_input = input("\n<user>")

    if user_input.lower() == 'exit':
        break

    conversation += f"user {user_input}\n"

    context = torch.tensor([encode(conversation)], dtype=torch.long, device=device)

    output = model.generate(context, max_new_tokens)

    generate = decode(output[0].tolist())

    response = generate[len(conversation):]

    if "<user>" in response:
        response = response.split("<user>")[0]

    print('', response)

    conversation += response + "\n"