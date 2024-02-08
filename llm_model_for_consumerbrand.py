import re
import torch
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.utils.data import Dataset, DataLoader

nltk.download('stopwords')
nltk.download('punkt')

with open('llm_model_training_data.txt', 'r', encoding ='utf-8') as file:
  text_data = file.read()

def preprocess_text(input_data):
  input_data = re.sub(r'[^a-zA-Z\s£$%+=<>]', '', input_data)
  tokens = word_tokenize(input_data)
  stop_words = set(stopwords.words('english'))
  tokens = [word for word in tokens if word.lower() not in stop_words]
  processed_data = ' '.join(tokens)
  return processed_data

preprocessed_data = preprocess_text(text_data)

class CompanyDataset(Dataset):
  def __init__(self, data, tokenizer, max_length=128):
    self.data = data
    self.tokenizer = tokenizer
    self.max_length = max_length
  def __len__(self):
    return len(self.data)
  def __getitem__(self, idx):
    input_text = self.data[idx]
    inputs = self.tokenizer(input_text, return_tensors="pt", max_length=self.max_length, truncation=True)
    inputs['input_ids'] = inputs['input_ids'].squeeze()
    return inputs

model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
company_dataset = CompanyDataset(data=preprocessed_data.split('.'), tokenizer=tokenizer)

batch_size = 8
dataloader = DataLoader(company_dataset, batch_size=batch_size)
model = GPT2LMHeadModel.from_pretrained(model_name)

epochs = 8
learning_rate = 1e-4
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
  for batch in dataloader:
    optimizer.zero_grad()
    outputs = model(**batch, labels=batch['input_ids'])
    loss = outputs.loss
    loss.backward()
    optimizer.step()

model.save_pretrained("new_fine_tuned_llm_model")
tokenizer.save_pretrained("new_fine_tuned_llm_model")

fine_tune_model = "new_fine_tuned_llm_model"
model = GPT2LMHeadModel.from_pretrained(fine_tune_model)

def generate_text(prompt, model, tokenizer, max_length=100, num_return_sequences=1, company_name=""):
  full_prompt = company_name + prompt
  input_ids = tokenizer.encode(full_prompt, return_tensors="pt")
  attention_mask = torch.ones(input_ids.shape, device=input_ids.device)
  output = model.generate(
      input_ids,
      attention_mask=attention_mask,
      max_length=max_length,
      num_return_sequences=num_return_sequences,
      no_repeat_ngram_size=2,
      top_k=50,
      top_p=0.95,
      temperature=0.7,
      do_sample=True
    )
  generated_texts = [tokenizer.decode(seq, skip_special_tokens=True) for seq in output]
  return generated_texts

company_name = "unilever"
prompt = "What is the company culture?"
generated_texts = generate_text(prompt, model, tokenizer, max_length=200, company_name=company_name)

# Print generated text
for i, generated_text in enumerate(generated_texts):
  print(f"This text is generated for: {generated_text}")