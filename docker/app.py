import streamlit as st
from pathlib import Path
import pandas as pd
import torch
import numpy as np
from datasets import load_dataset
import evaluate
import transformers
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from recursive_summary import main_summarizer

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATASET = 'dmacres/mimiciii-hospitalcourse-meta'

def main():
        
    st.set_page_config(
        page_title="Discharge Summary Generator",
        page_icon="🔎",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.title('Brief Hospital Course Discharge Summary Generation')

    if "example" not in st.session_state:
        st.session_state["example"] = {}
    
    tab1, tab2, tab3 = st.tabs(["Brief Hospital Course Generator", "Note Investigation", "Technologies"])

    tab1_dd = read_markdown_file('dropdowns/BHCG_dropdown.md')
    tab2_dd = read_markdown_file('dropdowns/NI_dropdown.md')
    
    metric = load_rouge()
    dataset = load_meta_dataset()
    
    with tab1:
        st.title("Select the Methods for Brief Hospital Course Text Generation")

        with st.expander('Click here to learn how to use the app...'):
            st.markdown(tab1_dd)
    
        note_num = st.number_input('Select which example to use for generation.', min_value = 1, max_value = 5357, value = 1, step = 1, key = 'example-selector')
        
        example = load_example_from_number(dataset, note_num)
        st.session_state['example']['note_num'] = note_num
        st.session_state['example']['data'] = example[0]
        
        # method = st.selectbox('Choose the method.', ('Recursive', 'fine-tuned google/pegasus-large', 'fine-tuned facebook/bart-large'), key ='method-selector')
        model_selection = st.selectbox('Choose the model.', ('google/pegasus-large', 'facebook/bart-large'), key ='model-selector')
        
        method = st.selectbox('Choose the summarization method.', ('Summary generation using extractive summaries', 'Recursive summarization on patient notes'), key ='method-selector')

        max_chunk_size = None

        if 'Recursive' in method:

            c1, c2 = st.columns([1,1], gap = 'medium')
    
            max_chunk_size = c1.slider('Select the maximum chunk size.', min_value = 128, max_value = 1024, value = 1024, key = 'chunkSize')
            target_len = c2.slider('Select the maximum summary length.', min_value = 128, max_value = 512, value = 512, key = 'RecTargetLen')
        else:
            target_len = st.slider('Select the maximum summary length.', min_value = 128, max_value = 1024, value = 512, key = 'BaseTargetLen')


        with st.container():
            
            generate_button = st.button('Run')
    
            if generate_button:

                target_text = example[0]['target_text']
                st.markdown('## Target Text')
                st.write(target_text)

                with st.spinner(text="Generating... This could take a while..."):
                    tokenizer, model = get_model_and_tokenizer(model_selection)
    
                    summary = summarize(method, example, tokenizer, model, max_chunk_size = max_chunk_size, target_len = target_len)
                    st.markdown('## Summary Text')
                    st.write(summary)

                    score = score_summ(metric, summary, target_text)
                    st.markdown('## ROUGE Score')
                    st.write(score)
                        

    with tab2:
        with st.container():
            
            st.title("Investigate Notes")
            
            with st.expander("Click here to learn about the options..."):
                st.markdown(tab2_dd)
                
            st.markdown(f"## Selected Note Number: {st.session_state['example']['note_num']}\n")
            text_category = st.selectbox('What would you like to view?', ('Target Text', 'Cosine Similarity Extractive Note Summary', 'Notes'), key = 'text-selector')

            key_mapper = {'Target Text': 'target_text', 'Cosine Similarity Extractive Note Summary': 'extractive_notes_summ', 'Notes': 'notes'}
            key_mapping = key_mapper[text_category]

            if key_mapping in ('target_text', 'extractive_notes_summ'):
                st.write(st.session_state['example']['data'][key_mapping])
            else:
                note_selector = st.selectbox('Which note would you like to view?', list(range(1, st.session_state['example']['data']['n_notes']+1)), key = 'note-selector')
                meta_note = st.session_state['example']['data'][key_mapping][note_selector-1]
                c1, c2, c3 = st.columns([1,1,1], gap = 'medium')
                c1.markdown(f"### CATEGORY:\n ###### {meta_note['category']}")
                c2.markdown(f"### DESCRIPTION:\n ###### {meta_note['description']}")
                c3.markdown(f"### CHART DATE:\n ###### {meta_note['chartdate']}")
                st.markdown(f"### NOTE TEXT:\n {meta_note['text']}")

        with tab3:
            with st.container():
                st.markdown("### Technologies")
                st.markdown(" ")
                
                st.markdown("##### The MIMIC-III Dataset")
                st.markdown(
                    """
                [The Medical Information Mart for Intensive Care version III (MIMIC-III)](https://physionet.org/content/mimiciii/1.4/) is a large, freely-available database comprising deidentified health-related data associated with over forty thousand patients who stayed in critical care units of the Beth Israel Deaconess Medical Center between 2001 and 2012. The database includes information such as demographics, vital sign measurements made at the bedside (~1 data point per hour), laboratory test results, procedures, medications, caregiver notes, imaging reports, and mortality (including post-hospital discharge).
                """
                )
    
                st.markdown("##### Hugging Face Hub 🤗")
                st.markdown(
                    """
                [The Hugging Face Hub](https://huggingface.co/)  is a platform with over 120k models, 20k datasets, and 50k demos in which people can easily collaborate in their ML workflows. The Hub works as a central place where anyone can share, explore, discover, and experiment with open-source Machine Learning..
                """
                )
                
                st.markdown("##### Streamlit")
                st.markdown(
                    """
                [Streamlit](https://streamlit.io/) is an open-source app framework for Machine Learning and Data Science teams. Streamlit turns data scripts into shareable web apps in minutes. All in pure Python. No front-end experience required.
                """
                )

                st.markdown("##### GitHub Repo")
                st.markdown(
                    """
                [Here](https://github.com/daltonmacs99/generate-discharge-summaries-mimiciii) is the link to the GitHub Repo for this app.
                """
                )


@st.cache_data(show_spinner=False)
def read_markdown_file(markdown_file):
    return Path(markdown_file).read_text()
    
@st.cache_data(show_spinner=False)
def load_rouge():
    return evaluate.load('rouge')


@st.cache_data(show_spinner=False)
def load_example_from_number(_dataset, num):
    return _dataset.select([num-1])


@st.cache_data(show_spinner=False)
def load_meta_dataset():
    return load_dataset(DATASET, split = 'test')


@st.cache_data(show_spinner=False)
def get_model_and_tokenizer(selection):
    return AutoTokenizer.from_pretrained(selection), AutoModelForSeq2SeqLM.from_pretrained(selection).to(DEVICE)


# @st.cache_data(show_spinner=False)
def summarize(method, _example, _tokenizer, _model, max_chunk_size = 1024, target_len = 512):

    if 'Recursive' in method:
        # TODO: make params configurable
        return main_summarizer(_example, _model, _tokenizer, max_chunk_size=max_chunk_size, target_len=target_len)

    else:
        inputs = _tokenizer(_example[0]['extractive_notes_summ'], max_length=1024,  truncation=True, 
                        padding="max_length", return_tensors="pt")
        
        summary = _model.generate(input_ids=inputs["input_ids"].to(DEVICE),
                         attention_mask=inputs["attention_mask"].to(DEVICE), 
                         length_penalty=0.8, num_beams=8, max_length=target_len).squeeze(0)
        
        decoded_summary = _tokenizer.decode(summary, skip_special_tokens=True, clean_up_tokenization_spaces=True) 

        return decoded_summary


@st.cache_data(show_spinner=False)
def score_summ(_metric, gen_sum, target_sum):
    score = _metric.compute(predictions = [gen_sum], references = [target_sum])
    return pd.DataFrame(score, index = [0])

    
if __name__ == '__main__':
    main()
