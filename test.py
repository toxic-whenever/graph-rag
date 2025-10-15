import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain_ollama import OllamaEmbeddings
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from neo4j import GraphDatabase
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars


def main():
    # load env variable
    load_dotenv()

    print("Đang tải và phân đoạn tài liệu...")
    loader = TextLoader(file_path="C:\\Users\\Admin\\PycharmProjects\\chatbot-source\\dummy-text.txt")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=24)
    documents = text_splitter.split_documents(documents=docs)
    print(f"Đã tạo {len(documents)} tài liệu được phân đoạn.")

    # --- 2. Thiết lập LLM và Graph Transformer ---
    print("Đang thiết lập LLM và Graph Transformer...")
    llm = OllamaFunctions(model="llama3.1:8b", temperature=0, format="json")
    llm_transformer = LLMGraphTransformer(llm=llm)

    # --- 3. Chuyển đổi tài liệu thành biểu đồ và nhập vào Neo4j ---
    print("Đang chuyển đổi tài liệu thành các nút và mối quan hệ của biểu đồ...")
    # graph_documents = llm_transformer.convert_to_graph_documents(documents)

    print("Đang thêm các tài liệu biểu đồ vào Neo4j...")
    graph = Neo4jGraph()
    # graph.add_graph_documents(
    #     graph_documents,
    #     baseEntityLabel=True,
    #     include_source=True
    # )
    print("Đã nhập thành công dữ liệu biểu đồ vào Neo4j.")

    # --- 4. Tạo chỉ mục Full-Text trong Neo4j ---
    # Điều này là cần thiết để truy vấn các thực thể một cách hiệu quả.
    try:
        print("Đang tạo chỉ mục full-text trong Neo4j...")
        driver = GraphDatabase.driver(
            uri=os.environ["NEO4J_URI"],
            auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"])
        )
        with driver.session() as session:
            session.run(
                """
                CREATE FULLTEXT INDEX `fulltext_entity_id` IF NOT EXISTS
                FOR (n:__Entity__)
                ON EACH [n.id];
                """
            )
        driver.close()
        print("Đã tạo chỉ mục full-text thành công.")
    except Exception as e:
        print(f"Lỗi khi tạo chỉ mục full-text (có thể đã tồn tại): {e}")

    # --- 5. Thiết lập Truy xuất vectơ ---
    print("Đang thiết lập truy xuất vectơ với các nhúng...")
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    vector_index = Neo4jVector.from_existing_graph(
        embeddings,
        search_type="hybrid",
        node_label="Document",
        text_node_properties=["text"],
        embedding_node_property="embedding"
    )
    vector_retriever = vector_index.as_retriever()
    print("Đã thiết lập thành công trình truy xuất vectơ.")

    # --- 6. Thiết lập Chuỗi trích xuất thực thể ---
    # Xác định cấu trúc để trích xuất thực thể từ câu hỏi
    class Entities(BaseModel):
        """Xác định thông tin về các thực thể."""
        names: list[str] = Field(
            ...,
            description="Tất cả các thực thể người, tổ chức hoặc doanh nghiệp xuất hiện trong văn bản",
        )

    # Tạo một chuỗi để trích xuất các thực thể được cấu trúc từ một câu hỏi
    entity_chain = llm.with_structured_output(Entities)

    # --- 7. Định nghĩa các hàm Truy xuất ---
    def graph_retriever(question: str) -> str:
        """
        Truy xuất các mối quan hệ lân cận của các thực thể được đề cập trong câu hỏi.
        """
        print(f"\nĐang truy xuất biểu đồ cho câu hỏi: '{question}'")
        result = ""
        try:
            entities = entity_chain.invoke(question)
            print(f"Các thực thể được trích xuất: {entities.names}")
            for entity in entities.names:
                response = graph.query(
                    """
                    CALL db.index.fulltext.queryNodes('fulltext_entity_id', $query, {limit:2})
                    YIELD node, score
                    CALL {
                      WITH node
                      MATCH (node)-[r:!MENTIONS]->(neighbor)
                      RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
                      UNION ALL
                      WITH node
                      MATCH (node)<-[r:!MENTIONS]-(neighbor)
                      RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS output
                    }
                    RETURN output LIMIT 50
                    """,
                    {"query": entity},
                )
                result += "\n".join([el['output'] for el in response])
            return result
        except Exception as e:
            print(f"Không thể truy xuất biểu đồ: {e}")
            return ""

    def full_retriever(question: str):
        """
            Kết hợp dữ liệu từ cả trình truy xuất biểu đồ và trình truy xuất vectơ.
        """
        print("Đang thực hiện truy xuất kết hợp (biểu đồ + vectơ)...")
        graph_data = graph_retriever(question)
        vector_data = [el.page_content for el in vector_retriever.invoke(question)]
        final_data = f"""Dữ liệu biểu đồ: {graph_data} Dữ liệu vectơ: {"#Document ".join(vector_data)} """
        return final_data

    # --- 8. Tạo Chuỗi RAG cuối cùng ---
    print("\nĐang tạo chuỗi RAG cuối cùng...")
    template = """Trả lời câu hỏi chỉ dựa trên ngữ cảnh sau: \n{context}\n\n Câu hỏi: {question}\nSử dụng ngôn ngữ tự nhiên và ngắn gọn.\nTrả lời:"""
    prompt = ChatPromptTemplate.from_template(template)

    chain = (
            {
                "context": full_retriever,
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
            | StrOutputParser()
    )
    #
    # # --- 9. Chạy Chuỗi với một câu hỏi ví dụ ---
    question = "How do I pay for customer support chat automatically?"
    print(f"\nĐang gọi chuỗi với câu hỏi: '{question}'")
    response = chain.invoke(input=question)
    print("\n--- Câu trả lời cuối cùng ---")
    print(response)
    print("-------------------------\n")


if __name__ == "__main__":
    main()
