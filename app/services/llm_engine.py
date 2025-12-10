import os
import json
from groq import Groq
from dotenv import load_dotenv

# Load API Key
load_dotenv()

# Inisialisasi Client Groq
client = Groq(
    api_key=os.getenv("GROQ_API_KEY"),
)

# Model yang Anda pilih
MODEL_NAME = "llama-3.3-70b-versatile"

class LLMEngine:
    
# ... (kode inisialisasi client Groq tetap sama) ...

    @staticmethod
    async def process_user_intent(user_text: str, available_skills: list[str]):
        skills_str = ", ".join(available_skills)
        
        system_prompt = f"""
        ROLE: Klasifikator Niat (Intent Classifier).
        SKILL TERSEDIA: {skills_str}
        
        TUGAS: 
        1. Tentukan kategori aksi.
        2. Deteksi SEMUA skill yang disebutkan user.
        
        OUTPUT JSON:
        {{
            "action": "START_EXAM" | "GET_RECOMMENDATION" | "CASUAL_CHAT",
            "detected_skills": ["Skill A", "Skill B"] (List of strings, kosongkan jika tidak ada)
        }}
        """
        
        user_prompt = f'Input: "{user_text}"'
        
        try:
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model=MODEL_NAME,
                temperature=0.1, # Sangat kaku
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"Error intent: {e}")
            return {"action": "CASUAL_CHAT", "detected_skill": None}

    @staticmethod
    async def casual_chat(user_text: str):
        """Untuk ngobrol santai jika tidak ada action khusus"""
        system_prompt="""
        ROLE: Kamu adalah asisten belajar bernama MORA.
        
        TUGAS:
        1. Hanya jawab pertanyaan terkait belajar pemrograman, AI, dan web development.
        
        """
        try:
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "Kamu adalah asisten belajar bernama MORA. Jawab ramah dan singkat."},
                    {"role": "user", "content": user_text}
                ],
                model=MODEL_NAME,
            )
            return response.choices[0].message.content
        except:
            return "Maaf saya sedang error."
            
    # ... (Fungsi generate_question dan evaluate_answer biarkan tetap ada) ...

    @staticmethod
    async def generate_question(topics: list[str], level: str):
        topics_str = ", ".join(topics)
        
        # System Prompt: Instruksi Dasar
        system_prompt = f"""
        ROLE: Kamu adalah Senior AI Engineer & Penguji Ujian Teknis.
        TARGET LEVEL: {level}
        TOPICS: {topics_str}
        
        TUGAS:
        Buatlah SATU soal studi kasus integrasi (gabungan) untuk menguji pemahaman kandidat.
        
        INSTRUKSI KHUSUS:
        1. Jangan menanyakan definisi. Buat skenario nyata.
        2. Output HARUS dalam format JSON valid.
        3. Pertanyaan hanya seputar What, How, Why
        """
        
        # User Prompt: Trigger Generasi
        user_prompt = """
        Buatkan soal beserta rubrik penilaiannya sekarang.
        
        OUTPUT FORMAT (JSON):
        {
            "question_text": "Teks pertanyaan untuk user...",
            "grading_rubric": {
                "key_concept": "Konsep utama...",
                "must_have_keywords": ["keyword1", "keyword2"],
                "explanation_focus": "Fokus penjelasan..."
            }
        }
        """
        
        try:
            # Panggil API Groq
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model=MODEL_NAME,
                temperature=0.7,
                # FITUR PENTING: Memaksa output JSON
                response_format={"type": "json_object"} 
            )
            
            # Parse string JSON ke Dictionary Python
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            print(f"Error generating question via Groq: {e}")
            return None

    @staticmethod
    async def evaluate_answer(user_answer: str, question_context: dict):
        rubric = question_context.get('grading_rubric')
        question = question_context.get('question_text')
        
        system_prompt = f"""
        ROLE: Kamu adalah Penilai Ujian (Grader) yang objektif.
        
        KONTEKS SOAL: "{question}"
        
        RUBRIK (KUNCI JAWABAN):
        - Konsep: {rubric['key_concept']}
        - Keyword Wajib: {rubric['must_have_keywords']}
        - Fokus: {rubric['explanation_focus']}
        
        TUGAS:
        Nilai jawaban user. Analisis maknanya secara semantik.
        Output HARUS JSON.
        """
        
        user_prompt = f"""
        JAWABAN USER: "{user_answer}"
        
        Berikan penilaianmu dalam format JSON berikut:
        {{
            "is_correct": boolean,
            "score": integer (0-100),
            "feedback": "Penjelasan singkat (maks 2 kalimat) kenapa benar/salah",
            "missing_concepts": ["list konsep yang kurang"]
        }}
        """
        
        try:
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model=MODEL_NAME,
                temperature=0.5, # Lebih rendah agar penilaian konsisten
                response_format={"type": "json_object"}
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            print(f"Error evaluating answer via Groq: {e}")
            # Return nilai default agar tidak crash
            return {"is_correct": False, "score": 0, "feedback": "Terjadi kesalahan sistem evaluasi.", "missing_concepts": []}
        
    @staticmethod
    async def analyze_psych_answer(user_answer: str, question_data: dict):
        """
        Menentukan user condong ke Opsi A atau B berdasarkan ketikan mereka.
        """
        prompt = f"""
        ROLE: Psikolog Penjurusan Karir IT.
        
        PERTANYAAN: "{question_data['question']}"
        OPSI A: "{question_data['options']['A']}"
        OPSI B: "{question_data['options']['B']}"
        
        JAWABAN USER: "{user_answer}"
        
        TUGAS:
        Analisis jawaban user. Apakah makna kalimatnya lebih dekat ke Opsi A atau Opsi B?
        
        OUTPUT JSON:
        {{
            "choice": "A" | "B",
            "reason": "Alasan singkat kenapa masuk kategori itu"
        }}
        """
        
        try:
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=MODEL_NAME,
                temperature=0.1, # Harus tegas
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            return {"choice": "A", "reason": "Error, default ke A"}

# Instance global
llm_engine = LLMEngine()