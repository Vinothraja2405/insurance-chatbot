import streamlit as st
import os
import nltk
from thirdai import licensing, neural_db as ndb
from openai import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI as LangChainOpenAI

# Download NLTK data
nltk.download("punkt")

# Licensing and setup
from dotenv import load_dotenv
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
thirdai_license_key = os.getenv("THIRDAI_LICENSE_KEY")

if thirdai_license_key:
    licensing.activate(thirdai_license_key)

openai_client = OpenAI(api_key=openai_api_key)
db = ndb.NeuralDB()
insertable_docs = []

# Add your document paths
doc_files =doc_files = [r"C:\Users\ADMIN\Desktop\data\accidental-death-benefit-rider-brochure.pdf",
r"C:\Users\ADMIN\Desktop\data\cash-back-plan-brochuree.pdf",
r"C:\Users\ADMIN\Desktop\data\gold-brochure (1).pdf",
r"C:\Users\ADMIN\Desktop\data\gold-brochure.pdf",
r"C:\Users\ADMIN\Desktop\data\guaranteed-protection-plus-plan-brochure.pdf",
r"C:\Users\ADMIN\Desktop\data\indiafirst-csc-shubhlabh-plan-brochure.pdf",
r"C:\Users\ADMIN\Desktop\data\indiafirst-life-elite-term-plan-brochure.pdf",r"C:\Users\ADMIN\Desktop\data\indiafirst-life-guaranteed-benefit-plan-brochure1.pdf",r"C:\Users\ADMIN\Desktop\data\indiafirst-life-insurance-khata-plan-brochure.pdf",r"C:\Users\ADMIN\Desktop\data\indiafirst-life-little-champ-plan-brochure.pdf",r"C:\Users\ADMIN\Desktop\data\indiafirst-life-long-guaranteed-income-plan-brochure.pdf",r"C:\Users\ADMIN\Desktop\data\indiafirst-life-micro-bachat-plan-brochure.pdf",
r"C:\Users\ADMIN\Desktop\data\indiafirst-life-plan-brochure.pdf",r"C:\Users\ADMIN\Desktop\data\indiafirst-life-radiance-smart-investment-plan-brochure.pdf",r"C:\Users\ADMIN\Desktop\data\indiafirst-life-saral-bachat-bima-plan-brochure.pdf",r"C:\Users\ADMIN\Desktop\data\indiafirst-life-saral-jeevan-bima-brochure.pdf",r"C:\Users\ADMIN\Desktop\data\indiafirst-life-smart-pay-plan-brochure.pdf",
r"C:\Users\ADMIN\Desktop\data\indiafirst-maha-jeevan-plan-brochure.pdf",
r"C:\Users\ADMIN\Desktop\data\indiafirst-money-balance-plan-brochure.pdf",
r"C:\Users\ADMIN\Desktop\data\indiafirst-pos-cash-back-plan-brochure.pdf",
r"C:\Users\ADMIN\Desktop\data\indiafirst-simple-benefit-plan-brochure.pdf",
r"C:\Users\ADMIN\Desktop\data\smart-save-plan-brochure.pdf",
r"C:\Users\ADMIN\Desktop\data\single-premium-brochure.pdf",
r"C:\Users\ADMIN\Desktop\data\tulip-brochure.pdf",
r"C:\Users\ADMIN\Desktop\data\wealth-maximizer-brochure.pdf"]

for file in doc_files:
    doc = ndb.PDF(file)
    insertable_docs.append(doc)
db.insert(insertable_docs, train=False)

# Initialize memory for conversation
memory = ConversationBufferMemory()

# Define prompt template
prompt_template = PromptTemplate(
    input_variables=["chat_history", "query", "context"],
    template=(
        "You are an insurance chatbot designed to provide information about insurance plans.\n"
        "Use the conversation history to provide better answers.\n\n"
        "Here is the previous conversation history: {chat_history}\n\n"
        "Context: {context}\n"
        "Question: {query}\n"
        "Answer in 3-4 lines. Ask if there is any additional doubt.\n"
    )
)

# Define LangChain OpenAI LLM wrapper
langchain_llm = LangChainOpenAI(api_key=openai_api_key)

# Conversation Chain
conversation_chain = ConversationChain(
    llm=langchain_llm,
    memory=memory,
    prompt=prompt_template,
)

def generate_answers(query, references):
    context = "\n\n".join(references[:3])
    inputs = {"query": query, "context": context, "chat_history": memory.buffer}
    return conversation_chain.run(inputs)

def generate_queries_chatgpt(original_query):
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates multiple search queries based on a single input query."},
            {"role": "user", "content": f"Generate multiple search queries related to: {original_query}"},
            {"role": "user", "content": "OUTPUT (5 queries):"}
        ]
    )
    generated_queries = response.choices[0].message.content.strip().split("\n")
    return generated_queries

def get_references(query):
    search_results = db.search(query, top_k=50)
    references = [result.text for result in search_results]
    return references

def reciprocal_rank_fusion(reference_list, k=60):
    fused_scores = {}
    for i in reference_list:
        for rank, j in enumerate(i):
            if j not in fused_scores:
                fused_scores[j] = 0
            fused_scores[j] += 1 / ((rank + 1) + k)
    reranked_results = {j: score for j, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)}
    return reranked_results

# Streamlit UI
if 'responses' not in st.session_state:
    st.session_state['responses'] = []

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

st.title("Insurance Bot")

if len(st.session_state['requests']) == 0 and len(st.session_state['responses']) == 0:
    st.session_state['responses'].append("How can I assist you with your insurance queries?")

user_input = st.text_input("Do you have any questions?", key="input")

if user_input:
    st.session_state['requests'].append(user_input)
    query_list = generate_queries_chatgpt(user_input)
    reference_list = [get_references(q) for q in query_list]
    r = reciprocal_rank_fusion(reference_list)
    ranked_reference_list = [i for i in r.keys()]
    ans = generate_answers(user_input, ranked_reference_list)
    st.session_state['responses'].append(ans)

# Display chat history
for i in range(len(st.session_state['responses'])):
    st.message(st.session_state['responses'][i], key=str(i) + '_bot')
    if i < len(st.session_state['requests']):
        st.message(st.session_state['requests'][i], is_user=True, key=str(i) + '_user')
