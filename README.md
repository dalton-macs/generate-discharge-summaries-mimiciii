# Generate the "Brief Hospital Course" section of patient discharge summaries

Generate a section of patient discharge summaries from the [MIMIC-III](https://mimic.mit.edu/docs/gettingstarted/) dataset using [Hugging Face](https://huggingface.co/) models. Extractive summarization via sentence embeddings, cosine similarity, and the PageRank algorithm was used to cut down a patient's notes to the model context size of 1024 tokens. These extracted summaries can then be used as input text for abstractive summarization using BART-Large or Pegasus-large. Fine tuning was also done with these extracted summaries and two models, however, ROUGE scores indicated decreased performance when compared to the base models (no fine-tuning). The streamlit app in the docker folder allows for interactive summary generation and note investigation. The app can be run through a conda environment, or by building and running the docker container (see steps below).

## Conda Environment Setup
1. Create the environment

        conda env create -f environment.yml
   
3. Activate the environment
   
        conda activate generate-discharge-summaries

### To run the notebooks associated with querying the MIMIC-III data using AWS Athena:
1. Request access to the dataset by following the steps [here](https://mimic.mit.edu/docs/gettingstarted/).
2. Run the CloudFormation stack by following the steps [here](https://aws.amazon.com/blogs/big-data/perform-biomedical-informatics-without-a-database-using-mimic-iii-data-and-amazon-athena/)
3. Create access keys for your AWS user profile
4. Add these access keys to the .env_template and rename the file to .env

## Running the streamlit app
### With Conda environment
1. Move into the docker directory
   
        cd docker
   
3. Run the streamlit app
   
        streamlit run app.py
   
5. Open localhost:8501 on a browser

### Running the streamlit app via Docker
#### Option 1: Building the image locally
1. Build the image
   
        docker build -t generate-discharge-summaries .
   
2. Run the container
   
        docker run -p 8501:8501 generate-discharge-summaries
   
3. Open localhost:8501 on a browser

#### Option 2: Pulling the image from Docker Hub
1. Pull the image

        docker pull dmacres99/generate-discharge-summaries

2. Run the container

        docker run -p 8501:8501 dmacres99/generate-discharge-summaries

3. Open localhost:8501 on a browser

This project is associated with the Masters of Data Science coursework (DS504) at Worcester Polytechnic Institute.

