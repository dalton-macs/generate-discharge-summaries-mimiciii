#### Example Selection
This app uses the test set (5,356 total examples) of a [custom dataset](https://huggingface.co/datasets/dmacres/mimiciii-hospitalcourse-meta/viewer/default/test) on Hugging Face Hub. The number selected corresponds to the example number in the dataset.

#### Model Selection
Select which model to use for the summarization task.
- [google/pegasus-large](https://huggingface.co/google/pegasus-large)
- [faceboook/bart-large](https://huggingface.co/facebook/bart-large)

#### Summarization Method Selection
Select which method to use for the summarization task.
- **Summary generation using extractive summaries**

  This method generates the brief hospital course summary by using the extractive summary as the input. Extractive summaries were done in efforts to create a single text where the total token length would fit inside the model context window of 1024 tokens. This was done using cosine similarities of embedded sentences followed by the PageRank algorithm to extract the most meaningful sentences from all notes.

  Fine tuning both the google/pegasus-large and the facebook/bart-large models was done with these extractive summaries. However, the ROUGE scores showed a performance decrease from the baseline non-fine tuned models. Therefore, only the base models are used in this app.
  
- **Recursive summarization on patient notes**

  This method generates the brief hospital course summary using a recursive summarization approach with the selected model. Each note is summarized using the model, the summaries are then combined, they are then batched based on the maximum chunk size, each batch is summarized, batches are combined, and the repeats until the summarization length is less than or equal to the maximum summary length.

#### Maximum Summary Length
The maximum number of tokens that the model can return. The actual summary length will not necessarily be this long, it just defines the upper limit.

#### Maximum Chunk Size (Recursive only)
The maximum chunk size for tokens in recursive summary. This value defines how many sentences to include in each batch. The lower the value, the higher the number of batches and therefore an increased number of individual summarizations. The higher the value, the lower number of batches and therefore decreased number of individual summarizations.

