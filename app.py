# Import required modules
from typing import Dict, Union
from span_marker import SpanMarkerModel
import streamlit as st
import json
from collections import defaultdict

# Initialize the SpanMarker model
model = SpanMarkerModel.from_pretrained("tomaarsen/span-marker-bert-base-fewnerd-fine-super").try_cuda()

def ner(text) -> Dict[str, Union[str, int, float]]:
    # Initialize a defaultdict to store the entities categorized by labels
    categorized_entities = defaultdict(list)
    
    for entity in model.predict(text):
        label = entity["label"]
        word = entity["span"]
        
        # Add the entity span to the appropriate label list
        categorized_entities[label].append(word)
    
    # Convert the defaultdict to a standard dict for JSON serialization
    categorized_entities = dict(categorized_entities)
    
    # Convert to JSON
    json_data = json.dumps(categorized_entities, indent=4)
    
    return json_data

# Streamlit app
st.title('Named Entity Recognition with SpanMarker')
st.write("Enter a text to identify named entities.")

# Text input
user_input = st.text_area("Input Text", "Amelia Earhart flew her single engine Lockheed Vega 5B across the Atlantic to Paris.")

# Button to run NER
if st.button('Run NER'):
    result = ner(user_input)
    st.write("## NER Result in JSON Format:")
    st.json(result)
  
