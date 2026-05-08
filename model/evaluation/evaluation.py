from transformers import AutoModel, AutoTokenizer

from model import MAX_DISTANCE, MAX_INPUT_LENGTH, MODEL_ID
from model.training.data_collator import LinkGramDataCollator
"""
Evaluation script

Script to load and test the final trained version of the model.
Can be used to collect more metrics apart from what is already collected
during training.
"""

#path to local model
LOCAL_PATH = './bart_linkgram_training/'
#ideally, add code to training to locally store the link_type dictionary and load that here
link_type_to_id = {}

"""
Load the trained model
"""
model = AutoModel.from_pretrained(LOCAL_PATH)

"""
Tokenizer
"""
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

"""
DataCollater
"""
data_collator = LinkGramDataCollator(tokenizer, MAX_INPUT_LENGTH, MAX_DISTANCE, link_type_to_id)
