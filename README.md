# MedQuery-Chatbot 💊🤖
With so much information floating around on the internet we don't know which information on the web should be believed and which shouldn't!
MedQuery chatbot is a LLM-based chatbot that gives medical advices users and helps them diagnose disease on the basis of symptoms straight from the medical books.<br>
..........<br>  
_To run the project the download the following libraries:_ <br>  
- pip install transformers <br>
- pip install langchain-community <br>
- pip install faiss-cpu <br>
- pip install chainlit <br>
- pip install sentence-transformers <br>
- pip install PyPDF2 <br>
- pip install torch <br>
..... <br>
Download the LLM Llama-2-7B-Chat-GGML from https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML <br>
..... <br>
DATA:
- https://www.zuj.edu.jo/download/gale-encyclopedia-of-medicine-vol-1-2nd-ed-pdf/
- https://emedicodiary.com/book/view/337/harrison-s-principles-of-internal-medicine#google_vignette
.....<br>
To run:<br>
Run the ingest file first to create a vectorstore and then run the model.
Use cmd and change the cmd directory to the directory location that contains the code files and run the command "chainlit run model.py -w".
