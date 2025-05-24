import os

from utils import *
from config import *
from prompt import *
from langchain.chains import LLMChain, LLMRequestsChain

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.vectorstores.chroma import Chroma
from langchain.vectorstores.faiss import FAISS
from langchain.schema import Document
from langchain.agents import ZeroShotAgent, AgentExecutor, Tool, create_react_agent
from langchain.memory import ConversationBufferMemory
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain import hub

from langchain_community.chat_models import ChatZhipuAI
#把keys设置在环境变量中
os.environ["ZHIPUAI_API_KEY"] = "73e8b921f04541469ff73ca558811385.wo1Jy4EjyRxUJzeu"
mdoel = ChatZhipuAI(
    model="glm-4-flash",
    temperature=0.9,
)
def structured_output_parser(response_schemas):
    text = '''
    请从以下文本中，抽取出实体信息，并按json格式输出，json包含首尾的 "```json" 和 "```"。
    以下是字段含义和类型，要求输出json中，必须包含下列所有字段：\n
    '''
    for schema in response_schemas:
        text += schema.name + ' 字段，表示：' + schema.description + '，类型为：' + schema.type + '\n'
    return text


NER_PROMPT_TPL = '''
1、从以下用户输入的句子中，提取实体内容。
2、注意：根据用户输入的事实抽取内容，不要推理，不要补充信息。

{format_instructions}
------------
用户输入：{query}
------------
输出：
'''
chroma = Chroma(
persist_directory=os.path.join(os.path.dirname(__file__), './data/db'),
embedding_function=get_embeddings_model()
)

def retrival_func(query):
    documents = chroma.similarity_search_with_score(query,k=5)
    #通过分过滤掉分数小于0.7的
    query_result = [doc[0].page_content for doc in documents if doc[1]>0.7]
    prompt = PromptTemplate.from_template(RETRIVAL_PROMPT_TPL)
    retrival_chain = LLMChain(
        llm = get_llm_model(),
        prompt=prompt,
        verbose=os.getenv('VERBOSE')
    )

    inputs = {
        'query': query,
        'query_result': '\n\n'.join(query_result) if len(query_result) else '没有查到'
    }
    return retrival_chain.invoke(inputs)['text']


def graph_func(query):
    #相应的模板
    response_schemas = [
        #表示一个输出字段的模板
        ResponseSchema(type='list',name='disease',description='疾病名称实体'),
        ResponseSchema(type='list',name='symptom',description='疾病症状实体'),
        ResponseSchema(type='list', name='drug', description='药品名称实体'),
    ]

    output_parser = StructuredOutputParser(response_schemas=response_schemas)
    # print(output_parser)


    #这是一个抽取模板    用于抽取实体
    format_instructions = structured_output_parser(response_schemas)

    ner_prompt = PromptTemplate(
        template=NER_PROMPT_TPL,
        partial_variables = {'format_instructions': format_instructions},
        input_variables = ['query']
    )
    # print(ner_prompt)

    ner_chain = LLMChain(
        llm = mdoel,
        prompt = ner_prompt,
        verbose = os.getenv('VERBOSE')
    )

    result = ner_chain.invoke({
        'query': query
    })['text']                  #{
                                    #   "disease": ["胃疼"],
                                    #   "symptom": ["胃疼"],
                                    #   "drug": []
                                    # }
    # print(result)

    ner_result = output_parser.parse(result)  #{'disease': ['胃疼'], 'symptom': ['胃疼'], 'drug': []}
    # print(ner_result)
    #使用实体识别的结果填充模板
    graph_templates = []

    for key,template in GRAPH_TEMPLATE.items():
        slot = template['slots'][0]
        slot_values = ner_result[slot]

        for value in slot_values:
            graph_templates.append({  # solt是插槽位置，value是'高血压'或'糖尿病']
                'question': replace_token_in_string(template['question'], [[slot, value]]),
                'cypher': replace_token_in_string(template['cypher'], [[slot, value]]),
                'answer': replace_token_in_string(template['answer'], [[slot, value]]),
            })
        if not graph_templates:
            return

    graph_documents = [
        Document(page_content=template['question'], metadata=template)
        for template in graph_templates
    ]

    db = FAISS.from_documents(graph_documents,get_embeddings_model())
    graph_documents_filter = db.similarity_search_with_score(query,k=3)

    #执行CQL
    query_result = []
    neo4j_conn = get_neo4j_conn()
    for document in graph_documents_filter:
        question = document[0].page_content
        cypher = document[0].metadata['cypher']
        answer = document[0].metadata['answer']
        try:                #执行CQL语句
            result = neo4j_conn.run(cypher).data()
            if result and any(value for value in result[0].values()):
                answer_str = replace_token_in_string(answer,list(result[0].items()))
                query_result.append(f'问题：{question}\n答案：{answer_str}')
        except:
            pass
    #总结答案
    prompt = PromptTemplate.from_template(GRAPH_PROMPT_TPL)
    graph_chain = LLMChain(
        llm=get_llm_model(),
        prompt=prompt,
        verbose=os.getenv('VERBOSE')
    )
    inputs = {
        'query': query,
        'query_result': '\n\n'.join(query_result) if len(query_result) else '没有查到'
        # 将查询到的转化为字符串，让大模型的回答只根据我们用知识图谱中查询的结果来回答
    }
    return graph_chain.invoke(inputs)['text']



def search_func(query):
    prompt = PromptTemplate.from_template(SEARCH_PROMPT_TPL)
    llm_chain = LLMChain(
        llm=get_llm_model(),
        prompt=prompt,
        verbose = os.getenv("VERBOSE")
    )
    llm_request_chain = LLMRequestsChain(
        llm_chain = llm_chain,
        request_key = 'query_result'    #这个是直接把查询之后的结果直接封装到prompt对应的参数中来
    )
    inputs = {
        'query':query,
        'url': 'https://www.google.com/search?q=' + query.replace(' ', '+')

    }
    return llm_request_chain.invoke(inputs)['output']


def query(query):
    tools = [
        Tool.from_function(
            name='generic_func',
            # func=lambda x: generic_func(x, query),   #在这里面我们没有定义generic_func()这个函数
            description='可以解答通用领域的知识，例如打招呼，问你是谁等问题',
        ),
        Tool.from_function(
            name="retrival_func",
            func=lambda x: retrival_func(x, query),
            description='用于回答寻医问药网相关问题',
        ),
        Tool(
            name='graph_func',
            func=lambda x: graph_func(x, query),
            description='用于回答疾病、症状、药物等医疗相关问题',
        ),
        Tool(
            name='search_func',
            func=search_func,
            description='其他工具没有正确答案时，通过搜索引擎，回答通用类问题',
        ),
    ]
    prompt = hub.pull('hwchase17/react-chat')
    prompt.template = '请用中文回答问题！Final Answer 必须尊重 Obversion 的结果，不能改变语义。\n\n' + prompt.template
    agent = create_react_agent(llm=get_llm_model(),tools=tools,prompt=prompt)
    #用户存储上面的两记忆
    memory = ConversationBufferMemory(memory_key = 'chat_history')
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        memory=memory,
        handle_parsing_errors=True,
        verbose=os.getenv('VERBOSE')
    )
    agent_executor.invoke({"input": query})['output']



if __name__ == '__main__':
    graph_func("今天有点胃疼和鼻炎，需要吃什么药物来进行缓解？")
    def query(self, query):
        tools = [
            Tool.from_function(
                name='generic_func',
                func=lambda x: self.generic_func(x, query),
                description='可以解答通用领域的知识，例如打招呼，问你是谁等问题',
            ),
            Tool.from_function(
                name='retrival_func',
                func=lambda x: self.retrival_func(x, query),
                description='用于回答寻医问药网相关问题',
            ),
            Tool(
                name='graph_func',
                func=lambda x: self.graph_func(x, query),
                description='用于回答疾病、症状、药物等医疗相关问题',
            ),
            Tool(
                name='search_func',
                func=self.search_func,
                description='其他工具没有正确答案时，通过搜索引擎，回答通用类问题',
            ),
        ]

        prefix = """请用中文，尽你所能回答以下问题。您可以使用以下工具："""
        suffix = """Begin!

        History: {chat_history}
        Question: {input}
        Thought:{agent_scratchpad}"""


        agent_prompt = ZeroShotAgent.create_prompt(
            tools=tools,
            prefix=prefix,
            suffix=suffix,
            input_variables=['inputs',"agent_scratchpad","chat_history"]
        )
        llm_chain = LLMChain(llm=get_llm_model(),prompt=agent_prompt)
        agent = ZeroShotAgent(llm_chain=llm_chain)

        #为了让模型能感知上面已经提到过的信息
        memory = ConversationBufferMemory(memory_key='chat_history')
        agent_chain = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=tools,
            memory=memory,
            verbose = os.getenv("VERBOSE")
        )
        print(agent_chain.invoke({"input": query})['output'])


