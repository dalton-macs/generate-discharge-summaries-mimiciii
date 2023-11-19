import math
import torch
from itertools import chain
import nltk
nltk.download('punkt')
from nltk import sent_tokenize
import re

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# DEVICE = 'cpu'


def split_dataset_notes(example, tokenizer):

    def tokenize_each_sentence(text):
        split_text = sent_tokenize(text)
        input_ids = []
        for sent in split_text:
            input_ids.append(tokenizer(sent, return_attention_mask = False, truncation = True)['input_ids'])

        return input_ids
    
    for idx, note in enumerate(example['notes']):
        tokenized_text_sentences = tokenize_each_sentence(note['text'])
        len_tok_text_sent = len(list(chain(*tokenized_text_sentences)))
        
        example['notes'][idx]['tokenized_text'] = tokenized_text_sentences
        example['notes'][idx]['tokenized_text_tot_len'] = len_tok_text_sent

    return example


def chunk_sentences(tokenized_text, chunk_size):

    chunk_size_counter = 0
    chunks_meta = []
    chunks = []
    for idx, sentence in enumerate(tokenized_text):
        
        chunk_size_counter += len(sentence)
        
        if idx == (len(tokenized_text)-1):
            if chunk_size_counter < chunk_size:
                chunks.append(sentence)
                chunks_meta.append(list(chain(*chunks)))
            else:
                chunks_meta.append(list(chain(*chunks)))
                chunks = []
                chunks.append(sentence)
                chunks_meta.append(list(chain(*chunks)))
            break

        if chunk_size_counter < chunk_size:
            # print('true')
            chunks.append(sentence)
            
        else:
            chunks_meta.append(list(chain(*chunks)))
            chunks = []
            chunks.append(sentence) # can't skip the sentence
            chunk_size_counter = len(sentence)

    return chunks_meta


def recursive_summarizer_meta(tokenized_text, model, tokenizer, max_chunk_size, target_len):

    def recursive_summarizer(tokenized_sentences, tot_len_store = [-3,-2,-1]):
        if DEVICE == 'cuda':
            torch.cuda.empty_cache()
        
        text_len = sum([len(sentence) for sentence in tokenized_sentences])
        tot_len_store.append(text_len)

        # prevent infinite recursion
        if (text_len<=target_len) or (tot_len_store[-1]==tot_len_store[-2]==tot_len_store[-3]):
            combined_sentences = list(chain(*tokenized_sentences))
            return tokenizer.decode(combined_sentences, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        n_chunks = math.ceil(text_len/max_chunk_size)
        chunk_size = math.floor(text_len/n_chunks)

        chunks = chunk_sentences(tokenized_sentences, chunk_size)

        chunk_summs = []
        for chunk in chunks:
            if len(chunk)==0:
                continue

            input_ids = torch.tensor(chunk).to(DEVICE).unsqueeze(0)
            attention_mask = torch.ones(input_ids.size()).to(DEVICE)

            chunk_summ = model.generate(input_ids=input_ids,attention_mask=attention_mask, length_penalty=0.8, num_beams=8, max_length=target_len)

            chunk_summs.append(chunk_summ)

        combined_tokenized_summ = torch.cat(chunk_summs, dim = 1).squeeze(0)
        decoded_combined_summ = tokenizer.decode(combined_tokenized_summ, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        # print(decoded_combined_summ)
        tokenized_combined_sentences = [tokenizer(sent, return_attention_mask = False, truncation = True)['input_ids'] for sent in sent_tokenize(decoded_combined_summ)]
        return recursive_summarizer(tokenized_combined_sentences, tot_len_store)

    return recursive_summarizer(tokenized_text)


def main_summarizer(orig_example, model, tokenizer, max_chunk_size, target_len):
    example = orig_example.map(split_dataset_notes, fn_kwargs = {'tokenizer': tokenizer})[0]
    note_meta_summ = []
    for idx, note in enumerate(example['notes']):
        
        if DEVICE == 'cuda':
            torch.cuda.empty_cache()
        
        # Recursively summarize note if tot len is longer than model context len
        if note['tokenized_text_tot_len']>max_chunk_size:
            # print(f'Note {idx} getting recursively summarized')
            note_summ = recursive_summarizer_meta(note['tokenized_text'], model, tokenizer, max_chunk_size, target_len)
            note_meta_summ.append(note_summ)
        # Otherwise summarize the note
        else:
            input_ids = torch.tensor(list(chain(*note['tokenized_text']))).to(DEVICE).unsqueeze(0)
            attention_mask = torch.ones(input_ids.size()).to(DEVICE)

            # Summarize each individual note
            note_summ_tokenized = model.generate(input_ids=input_ids,
                 attention_mask=attention_mask, 
                 length_penalty=0.8, num_beams=8, max_length=target_len).squeeze(0)

            note_summ = tokenizer.decode(note_summ_tokenized, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            
            note_meta_summ.append(note_summ)

    # combine note sums
    combined_note_summ = ' '.join(note_meta_summ)
    # print(f"Combined Ind Note Summs:\n\n{combined_note_summ}")

    # split into sentences and tokenize
    combined_note_summ_sent_tokenized = [tokenizer(sent, return_attention_mask = False)['input_ids'] for sent in sent_tokenize(combined_note_summ)]
    
    # run recursive_summarizer_meta
    final_summ = recursive_summarizer_meta(combined_note_summ_sent_tokenized, model, tokenizer, max_chunk_size, target_len)
    final_summ = re.sub('_+', ' ', final_summ)

    return final_summ
    