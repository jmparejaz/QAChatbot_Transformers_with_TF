import os
import warnings
from ontology_dc8f06af066e4a7880a5938933236037.simple_text import SimpleText

from openfabric_pysdk.context import OpenfabricExecutionRay
from openfabric_pysdk.loader import ConfigClass
from time import time
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForQuestionAnswering


tokenizer = AutoTokenizer.from_pretrained("jmparejaz/QA-finetuned-distilbert-TFv3")
model = AutoModelForQuestionAnswering.from_pretrained("jmparejaz/QA-finetuned-distilbert-TFv3",from_tf=True)
pipeline = pipeline('question-answering', model=model, tokenizer=tokenizer)



############################################################
# Callback function called on update config
############################################################
def config(configuration: ConfigClass):
    # TODO Add code here
    pass


############################################################
# Callback function called on each execution pass
############################################################
def execute(request: SimpleText, ray: OpenfabricExecutionRay) -> SimpleText:
    output = []
    
    for text in request.text:

        if not isinstance(text, dict) or 'context' not in text.keys() or 'questions' not in text.keys():
            raise ValueError("Invalid input format. Expecting a dictionary with 'context' and 'questions' as keys.")
    
        # Extract the context and questions from the input
        context = text['context']
        questions = text['questions']
        prediction=pipeline(context,questions)
        response = prediction['answer']
        output.append(response)

    return SimpleText(dict(text=output))

