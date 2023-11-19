**_The investigation in this tab is based on the selected example from the previous tab._**

#### Target Text
The Hospital Course/Brief Hospital Course section of the patient's discharge summary for their hospital visit. This is the target summarization the models/methods aim to generate.

#### Cosine Similarity Extractive Note Summary
Extractive summaries were done in efforts to create a single text where the total token length would fit inside the model context window of 1024 tokens. This was done using cosine similarities of embedded sentences followed by the PageRank algorithm to extract the most meaningful sentences from all notes.

#### Notes
The individual preprocessed notes of the patients. Select the note number (individual notes are sorted by obfuscated chart dates) to see the note text, category, description, and chart date.