from langchain.chains import LLMChain
#from langchain.text_splitter import CharacterTextSplitter
#from langchain.utilities import GoogleSearchAPIWrapper

from steamship.invocable import PackageService, get, post
from steamship_langchain.llms import OpenAI
from steamship_langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory

#from unstructured.documents.html import HTMLDocument

from prompt import CHATBOT_PROMPT

#emcrit_search = GoogleSearchAPIWrapper()

class DocBotPackage(PackageService):
    @post("/send_message")
    def send_message(self, message: str, chat_history_handle: str) -> str:
        """Does a search, uses that to find relevant content, passes that through a summarizer."""
        llm = OpenAI(client=self.client, temperature=0)
        
        #results = emcrit_search._google_search_results(message, num=3)
        #import requests
        #top_hit_text = requests.get(results[0]['formattedUrl']).text
        #top_hit_doc = HTMLDocument.from_string(top_hit_text)[0].__str__()
        #
        #text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        #texts = text_splitter.split_text(top_hit_doc)


        #if llm.get_num_tokens(top_hit_doc) > 1024:
            ## Truncate if long
            #top_hit_doc = top_hit_doc[:int(1024*4)]
        
        # steamship_memory will persist/retrieve conversation across API calls
        steamship_memory = ConversationBufferWindowMemory(
            client=self.client, key=chat_history_handle, k=2
        )
        chatgpt = LLMChain(
            llm=llm,
            prompt=CHATBOT_PROMPT,
            memory=steamship_memory,
        )
        return chatgpt.predict(human_input=message)

    @get("/transcript")
    def transcript(self, chat_history_handle: str) -> str:
        """Return the full transcript for a chat session."""

        # we can use the non-windowed memory to retrieve the full history.
        steamship_memory = ConversationBufferMemory(client=self.client, key=chat_history_handle)
        
        return steamship_memory.load_memory_variables(inputs={}).get("history", "")

