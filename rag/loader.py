from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_scripture(path="data/gita.txt"):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50
    )

    chunks = splitter.split_text(text)
    return chunks
