"""To run when you have to download the model to local.
We won't consider it while doing inference.
"""
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from constants import * 


tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
model = AutoModelForQuestionAnswering.from_pretrained(MODEL)

model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)