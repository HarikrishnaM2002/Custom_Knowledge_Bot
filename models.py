from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Save the model and tokenizer to a local folder
model_name = "t5-small"
save_path = "./local_models/t5-small"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(save_path)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model.save_pretrained(save_path)
