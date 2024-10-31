from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl

DB_FAISS_PATH = 'vectorstore/db_faiss'

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    print("Setting custom prompt...")
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])
    return prompt

def retrieval_qa_chain(llm, prompt, db):
    print("Creating RetrievalQA chain...")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=db.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )
    return qa_chain

def load_llm():
    print("Loading LLM...")
    llm = CTransformers(
        model="TheBloke/Llama-2-7B-Chat-GGML",
        model_type="llama",
        max_new_tokens=512,
        temperature=0.5
    )
    return llm

def qa_bot():
    print("Setting up QA bot...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)
    return qa

def final_result(query):
    print(f"Processing query: {query}")
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response

@cl.on_chat_start
async def start():
    print("Chat started...")
    chain = qa_bot()
    cl.user_session.set("chain", chain)
    cl.user_session.set("history", [])
    msg = cl.Message(content="Hi, Welcome to MedQuery ChatBot. What is your query?")
    await msg.send()

@cl.on_message
async def main(message: cl.Message):
    print(f"Received message: {message.content}")
    chain = cl.user_session.get("chain")
    history = cl.user_session.get("history")
    
    cb = cl.AsyncLangchainCallbackHandler(stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"])
    cb.answer_reached = True
    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["result"]
    sources = res["source_documents"]
    
    if sources:
        answer += f"\nSources:" + str(sources)
    else:
        answer += "\nNo sources found"
'''
    # Save the message and response to history
    history.append({"question": message.content, "answer": answer})
    cl.user_session.set("history", history)
    
    await cl.Message(content=answer).send()
    
    # Display history in sidebar
    sidebar_content = "\n\n".join([f"Q: {h['question']}\nA: {h['answer']}" for h in history])
    await cl.SidebarMessage(content=sidebar_content).send()
'''    

if __name__ == "__main__":
    print("Starting Chainlit server...")
    # Remove or comment out the cl.run() line if present
    # cl.run()
