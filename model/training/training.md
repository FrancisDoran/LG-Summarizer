
# Training Techniques

### TOC
1. [General Approach](#general-approach)
2. [Training Loop API via huggingface](#training-loop-api-via-huggingface)
3. [PEFT: Parameter-Efficient Fine-Tuning via LoRA](#peft-parameter-efficient-fine-tuning-via-lora)

#### General Approach

1. Define a model \(see model.py\)
2. Define a dataset
3. Define a [PEFT config](#peft-parameter-efficient-fine-tuning-via-lora)
    * Use target\_modules directed at the bias Embedding layers of the custom model
    * Use modules\_to\_save to save the link\_type\_dict along with the model.
4. Pass the model, and the PEFT config to the get\_peft\_model method to
get our model with a PEFT wrapper.
5. Define training arguments (TrainingArguments class)
6. Define a Trainer class

### Training Loop API via huggingface

This module allows for easy creation of training loops in the form of classes.

#### Key methods/classes:

* class: TrainingArguments
    * This class defines micro level training arguments.  
    * attribute: output\_dir
    * attribute: num\_train\_epochs
    * attribute: per\_device\_train\_batch\_size
    * attribute: per\_device\_eval\_batch\_size
    * attribute: logging\_dir
    * attribute: logging\_steps
    * attribute: evaluation\_strategy
    * attribute: save\_strategy
    * attribute: load\_best\_model\_at\_end
    * attribute: learning\_rate

* class: Trainer
    * This class defines the macro level training loop/params.
    * attribute: model
    * attribute: arguments
    * attribute: train\_dataset
    * attribute: eval\_dataset
    * method: train
    * method: evaluate
    * method: predict

### PEFT: Parameter-Efficient Fine-Tuning via LoRA

This module allows for a high level API to perform training on a model's  
extra parameters \(parameters not apart of the pre-trained model\)

[PEFT Basics](https://huggingface.co/docs/peft/quicktour)

#### Key methods/classes:

* class: LoraConfig  
    * attribute: target\_modules
    * attribute: modules\_to\_save
    * attribute: use\_dora

* method: get\_peft\_model
