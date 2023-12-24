import pickle
import torch
from exllamav2 import ExLlamaV2Tokenizer, ExLlamaV2Config

def load_distributions(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

def decode_distributions(distributions, tokenizer: ExLlamaV2Tokenizer):
    decoded_conversations = []
    for conversation in distributions:
        decoded_conversation = []
        for turn_distributions in conversation:
            turn_token_ids = []
            for distribution in turn_distributions:
                distribution_tensor = torch.tensor(distribution).to('cpu').float()
                probabilities = torch.softmax(distribution_tensor, dim=0)
                most_likely_token_id = torch.argmax(probabilities).item()
                turn_token_ids.append(most_likely_token_id)
            decoded_turn = tokenizer.decode(torch.tensor((turn_token_ids)))
            decoded_conversation.append(decoded_turn)
        decoded_conversations.append(decoded_conversation)
    return decoded_conversations

# Configuration and Tokenizer Initialization
model_path = r"C:\Users\gololo\Desktop\neural-chat-7b-v3-1-exl2"
config = ExLlamaV2Config()
config.model_dir = model_path
config.prepare()
tokenizer = ExLlamaV2Tokenizer(config)

# Load and decode distributions
file_path = "F:\\distilled\\janny_Filteredtest\\neural-chat-7b-v3-1-exl2\\distributions_1.pkl"
distributions = load_distributions(file_path)
decoded_conversations = decode_distributions(distributions, tokenizer)

# Display the structure and content of the decoded conversations
print(f"Total number of conversations in file: {len(decoded_conversations)}")
for i, conversation in enumerate(decoded_conversations):
    print(f"Conversation {i+1} (Turns: {len(conversation)}):")
    for f, turn in enumerate(conversation):
        print(f"  Turn {f+1}: {repr(turn)}")
    print("\n")  # Add a new line for better separation between conversations