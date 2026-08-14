class ResumeJobMatcher:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None  # sẽ load model transformer thật ở bước sau

    def load_model(self):
        # TODO: from sentence_transformers import SentenceTransformer
        # self.model = SentenceTransformer(self.model_name)
        pass

    def compute_similarity(self, resume_text: str, job_text: str) -> float:
        # TODO: thay bằng cosine similarity giữa 2 embedding thật
        # Hiện tại trả giá trị giả để test luồng API
        return 0.75

matcher = ResumeJobMatcher(model_name="all-MiniLM-L6-v2")