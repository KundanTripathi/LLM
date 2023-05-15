'''Pdf query with source and using openai to produce output on it's basis'''

import openai
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

load_dotenv()
os.environ['OpenAI_API_Key'] = os.getenv('OpenAI_API_Key')
reader = PdfReader(os.getenv('filepath'))

raw_text = ''
for i, page in enumerate(reader.pages):
    text = page.extract_text()
    if text:
        raw_text += text

#print(raw_text[:500])

text_splitter = CharacterTextSplitter(chunk_size= 500, chunk_overlap= 100, separator= "\n", length_function= len)
texts = text_splitter.split_text(raw_text)

embeddings = OpenAIEmbeddings()
docvecs = FAISS.from_texts(texts, embeddings)

chain = load_qa_chain(OpenAI(), chain_type="stuff")

query = "What are key Data Science Techniques and Technical Topics required for a Data Scientist to learn, understand and create project based on content in the text?"
docs = docvecs.similarity_search(query)
print(chain.run(input_documents = docs, question = query))

#print(docs[0])
#print(docs[1])
context=""
for i in range(len(docs)):
    if i<5:
        context += docs[i].page_content
print(context)

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain

# Initialling Model Package 
llm  =  OpenAI(temperature=0.7)

# Prompt 
output1 = PromptTemplate(
    input_variables= ["context"],
    template = "With provided context {context} design a 3 weeks Data Science technical training program for a Data Scientist"
    )

output2 = PromptTemplate(
    input_variables=["Training"],
    template = "for the {Training}, what are 5 best and chepest sources available that provides structure courses which covers all the topics "
    )

# Chain 
chain1 = LLMChain(llm=llm, prompt=output1,output_key="Hotels",verbose=True)
chain2 = LLMChain(llm=llm, prompt=output2, verbose=True)

final_out = SimpleSequentialChain(chains=[chain1,chain2], verbose=True)
print(final_out.run(context))


