import os
import json
from groq import AsyncGroq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class LLMEngine:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in .env file")
        
        self.client = AsyncGroq(api_key=self.api_key)

    async def process_user_intent(self, user_text: str, available_skills: list):
        # Ubah list skill jadi string biar AI tau menu apa aja yang ada
        skills_str = "\n".join([f"- {s}" for s in available_skills])
        
        system_prompt = f"""
        ROLE: Kamu adalah 'Router' untuk MORA, sebuah AI Learning Assistant.
        Tugasmu BUKAN menjawab pertanyaan, tapi mengarahkan user ke fitur yang benar.
        
        DAFTAR SKILL TERSEDIA DI DATABASE:
        {skills_str}
        
        INSTRUKSI UTAMA:
        Analisis pesan user dan tentukan ACTION JSON.
        
        1. ACTION: "START_EXAM"
            - Trigger: User ingin "tes", "ujian", "uji kemampuan", "soal", atau menyebut topik teknis (SQL, Python, CV, NLP).
            - TUGAS PENTING (MAPPING): User sering menyebut topik spesifik (misal "SQL"). Kamu WAJIB mencocokkannya dengan "Nama Skill Tersedia" yang paling relevan.
                Contoh: 
                - User: "Tes SQL" -> Detected: "Software & Data Foundations"
                - User: "Tes Vision" -> Detected: "Deep Learning & Computer Vision"
            
        2. ACTION: "GET_RECOMMENDATION"
            - Trigger: User minta "saran", "belajar apa", "rekomendasi", "bingung mulai mana".
            
        3. ACTION: "START_PSYCH_TEST"
            - Trigger: User tanya "karir", "cocok kerja apa", "tes minat".
            
        4. ACTION: "CASUAL_CHAT"
            - Trigger: Hanya untuk sapaan ("Halo"), curhat, atau pertanyaan di luar konteks belajar.
            - JANGAN gunakan ini jika user jelas-jelas minta tes/soal.
        
        OUTPUT JSON (Hanya JSON, tanpa teks lain):
        {{
            "action": "...",
            "detected_skills": ["Nama Skill Database 1", "Nama Skill Database 2"] (Array berisi String nama skill persis dari daftar diatas. Kosongkan jika tidak ada.)
        }}
        """
        
        try:
            chat_completion = await self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text}
                ],
                model="llama-3.3-70b-versatile", # Wajib model 70b biar pinter mapping
                temperature=0.0, # Wajib 0 agar tidak kreatif/halu
                response_format={"type": "json_object"}
            )
            
            response_content = chat_completion.choices[0].message.content
            print(f"DEBUG AI MAPPING: {response_content}") # Cek di terminal mappingnya bener gak
            return json.loads(response_content)
        except Exception as e:
            print(f"Error Intent: {e}")
            return {"action": "CASUAL_CHAT", "detected_skills": []}

    async def generate_question(self, topics: list, level: str):
        topics_str = ", ".join(topics)
        prompt = f"""
        Buatkan 1 soal esai pendek untuk menguji pemahaman user tentang topik: {topics_str}.
        Tingkat Kesulitan: {level}.
        Bahasa: Indonesia.
        
        Output JSON:
        {{
            "question_text": "Pertanyaan...",
            "grading_rubric": {{
                "keywords": ["kata1", "kata2"],
                "explanation_focus": "Poin utama yang harus dijelaskan"
            }}
        }}
        """
        try:
            completion = await self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                # UPDATE MODEL
                model="llama-3.3-70b-versatile",
                response_format={"type": "json_object"}
            )
            return json.loads(completion.choices[0].message.content)
        except:
            return None

    async def evaluate_answer(self, user_answer: str, question_context: dict):
        prompt = f"""
        Bertindaklah sebagai Dosen AI yang menilai jawaban mahasiswa.
        
        Soal/Konteks: {json.dumps(question_context)}
        Jawaban Mahasiswa: "{user_answer}"
        
        Tugas:
        1. Beri skor 0-100.
        2. Beri feedback singkat & ramah (Bahasa Indonesia).
        3. Tentukan apakah jawaban BENAR secara konsep (is_correct).
        
        Output JSON:
        {{
            "score": 85,
            "feedback": "Penjelasanmu bagus, tapi kurang detail di bagian...",
            "is_correct": true
        }}
        """
        try:
            completion = await self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                # UPDATE MODEL
                model="llama-3.3-70b-versatile",
                response_format={"type": "json_object"}
            )
            return json.loads(completion.choices[0].message.content)
        except:
            return {"score": 0, "feedback": "Error menilai.", "is_correct": False}

    async def casual_chat(self, user_text: str, history: list = [], syllabus_context: str = ""):
        
        prompt_template = f"""
        [ROLE]
        Namamu MORA. Kamu adalah Mentor & Asisten Teknis Spesialis.
        
        [PENGETAHUAN & SILABUS KAMU]
        Kamu HANYA menguasai materi yang tertera di bawah ini. Gunakan daftar ini sebagai acuan validasi jawabanmu:
        
        {syllabus_context}

        [TUGAS UTAMA]
        1. **JAWAB PERTANYAAN TEKNIS:**
            Jika user bertanya definisi atau konsep tentang topik yang ADA di silabus di atas (misal: "Apa itu List?", "Jelaskan Supervised Learning"), JAWABLAH dengan penjelasan konseptual yang singkat, padat, dan mudah dimengerti (maksimal 3-4 kalimat).
            
        2. **GAYA MENGAJAR:**
            Gunakan analogi sederhana jika perlu. Jangan terlalu kaku seperti buku teks, tapi tetap akurat.
            
        3. **BATASAN (BLACK LIST):**
            - Jika topik TIDAK ADA di silabus (misal: Hardware, IoT, Masak, Mobil, coding selain yang di sylabus), TOLAK dengan sopan dan pivot ke materi silabus.
            - JANGAN BERIKAN KODE FULL. Jika user minta "Buatkan kodingan", arahkan mereka untuk mengambil Ujian/Tes. Kamu hanya menjelaskan *Konsep* dan *Logika*.
            - Hindari jawaban yang terlalu panjang (lebih dari 4 kalimat).
            - Jika tidak yakin, katakan "Maaf, itu di luar pengetahuan saya."
            - Jangan buat-buat jawaban untuk topik di luar silabus.

        [CONTOH INTERAKSI]
        User: "Apa itu Supervised Learning?"
        Mora: (Cek silabus... ada!) "Supervised Learning itu ibarat belajar dengan kunci jawaban. Model dilatih pakai data yang sudah ada labelnya, jadi dia tau mana yang benar dan salah. Contohnya kayak filter email spam!"
        
        User: "Gimana cara bikin robot?"
        Mora: (Cek silabus... tidak ada!) "Waduh, itu ranah hardware/robotik, di luar silabusku. Tapi kalau kamu mau tau cara memprogram otak robotnya pakai Python (AI), aku bisa jelaskan logikanya!"

        [PERSONALITY]
        Ramah, Suportif, Emoji secukupnya.
        """

        system_msg = {
            "role": "system", 
            "content": prompt_template
        }
        
        messages = [system_msg]
        for msg in history[-5:]:
            messages.append({"role": msg['role'], "content": msg['content']})
        messages.append({"role": "user", "content": user_text})
        
        try:
            completion = await self.client.chat.completions.create(
                messages=messages,
                model="llama-3.3-70b-versatile", 
                temperature=0.3 
            )
            return completion.choices[0].message.content
        except Exception as e:
            return f"Maaf, otak saya sedang error. (Error: {str(e)})"
        

    async def analyze_psych_result(self, role: str, traits: list[str]):
        """
        Membuat penjelasan psikologis kenapa user cocok di role tersebut.
        """
        traits_str = "\n".join(traits)
        
        prompt = f"""
        Kamu adalah Konsultan Karir IT yang ahli membaca kepribadian.
        
        DATA USER:
        User baru saja mengikuti tes kepribadian sederhana.
        Hasil kecocokan tertinggi: **{role}**.
        Kebiasaan/Pilihan User:
        {traits_str}
        
        TUGAS:
        Berikan analisis singkat (maksimal 3 kalimat) dan memotivasi.
        Jelaskan hubungan antara kebiasaan user di atas dengan job role {role}.
        Gunakan gaya bahasa santai tapi meyakinkan.
        
        Contoh Output:
        "Wah, kamu punya bakat alami jadi AI Engineer! Kebiasaanmu yang suka menganalisis fakta dan mencari review mendalam menunjukkan kamu punya pola pikir analitis yang kuat, modal penting buat ngolah data!"
        """
        
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.1-8b-instant",
                temperature=0.7
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            return f"Berdasarkan jawabanmu, kamu sangat cocok menjadi {role}! Semangat belajar ya! 🚀"

llm_engine = LLMEngine()