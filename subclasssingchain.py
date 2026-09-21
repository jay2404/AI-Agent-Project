import os
from langchain_groq import ChatGroq

from langchain_classic.chains.base import Chain
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI




llm = ChatGroq(model="openai/gpt-oss-20b", api_key=os.getenv("GROQ_API_KEY"))

summary_prompt = PromptTemplate(
    input_variables=["text"],
    template="Summarize the following paragraph in one line:\n{text}"
)

translate_prompt = PromptTemplate(
    input_variables=["summary"],
    template="Translate the following English sentence into Hindi:\n{summary}"
)


class SummarizeTranslateChain(Chain):

    summarize_chain: object
    translate_chain: object

    @property
    def input_keys(self):
        return ["text"]

    @property
    def output_keys(self):
        return ["translated_text"]

    def _call(self, inputs):
        """Custom logic: summarize -> translate"""

        text = inputs["text"]

        summary = self.summarize_chain.invoke({
            "text": text
        })

        translated = self.translate_chain.invoke({
            "summary": summary
        })

        return {
            "translated_text": translated
        }


if __name__ == "__main__":

    summarize_chain = (
        summary_prompt
        | llm
        | StrOutputParser()
    )

    translate_chain = (
        translate_prompt
        | llm
        | StrOutputParser()
    )

    custom_chain = SummarizeTranslateChain(
        summarize_chain=summarize_chain,
        translate_chain=translate_chain
    )

    result = custom_chain.invoke({
        "text": (
            "Artificial Intelligence helps machines learn from data "
            "and make intelligent decisions, enabling them to perform "
            "tasks like understanding language, recognizing images, "
            "and predicting outcomes. It continues to evolve, making "
            "technology smarter and more adaptive to human needs."
        )
    });

    print("Final Output:\n", result["translated_text"])