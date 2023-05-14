''' In this we are using prompt templete and llmchain'''

import openai
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain

# Loading API Key
from dotenv import load_dotenv
load_dotenv()

import os 
os.environ['OpenAI_API_Key'] = os.getenv('OpenAI_API_Key')

# Initialling Model Package 
llm  =  OpenAI(temperature=0.7)

# Prompt 
output1 = PromptTemplate(
    input_variables= ["Cheapest_What"],
    template = " Tell us the cheapest of {Cheapest_What} in Santorini"
    )

output2 = PromptTemplate(
    input_variables=["Hotels"],
    template = "What is the price range this {Hotels} during June?"
    )

# Chain 
chain1 = LLMChain(llm=llm, prompt=output1,output_key="Hotels",verbose=True)
chain2 = LLMChain(llm=llm, prompt=output2, verbose=True)

final_out = SimpleSequentialChain(chains=[chain1,chain2], verbose=True)
print(final_out.run("Hotel"))

#print(chain1.run(Cheapest_What= "Hotel", Location= "Santorini"))
    
#print(output1.format(Cheapest_What= "Hotel", Location= "Santorini"))

#print(llm(output1.format(Cheapest_What= "Hotel", Location= "Santorini")))



