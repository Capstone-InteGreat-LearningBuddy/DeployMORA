import os
import json
from groq import AsyncGroq
from dotenv import load_dotenv

load_dotenv()

class LLMEngine:
    def __init__(self):
        self.clients = []
        
        key1 = os.getenv("GROQ_API_KEY")
        if key1:
            try:
                self.clients.append(AsyncGroq(api_key=key1))
            except Exception as e:
                print(f"⚠️ Gagal memuat Token Utama: {e}")

        key2 = os.getenv("GROQ_API_KEY_BACKUP")
        if key2:
            try:
                self.clients.append(AsyncGroq(api_key=key2))
            except Exception as e:
                print(f"⚠️ Gagal memuat Token Backup: {e}")
            
        print(f"✅ LLM Engine (Async) siap dengan {len(self.clients)} Client aktif.")

    async def _execute_with_retry(self, messages, model, temperature=0.5, response_format=None):
        """
        Mencoba request Async secara bergantian.
        """
        if not self.clients:
            raise Exception("Tidak ada API Key Groq yang terdeteksi di .env!")

        last_error = Exception("Unknown Error")

        for i, client in enumerate(self.clients):
            try:
                completion = await client.chat.completions.create(
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    response_format=response_format
                )
                return completion.choices[0].message.content

            except Exception as e:
                print(f"⚠️ Token ke-{i+1} Gagal. Error: {e}")
                last_error = e
                continue
        
        print("❌ Semua Token Gagal/Habis.")
        raise last_error

    async def process_user_intent(self, user_text: str, available_skills: list):
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
            response_content = await self._execute_with_retry(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text}
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            return json.loads(response_content)
        except Exception as e:
            print(f"Error Router: {e}")
            return {"action": "CASUAL_CHAT", "detected_skills": []}

    async def generate_question(self, topics: list, level: str):
        topics_str = ", ".join(topics)
        prompt = f"""
        Buatkan 1 soal esai pendek dengan konsep how, what, why untuk menguji pemahaman user
        Tentang topik: {topics_str}.
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
            response_content = await self._execute_with_retry(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile",
                temperature=0.5,
                response_format={"type": "json_object"}
            )
            return json.loads(response_content)
        except Exception as e:
            print(f"ERROR Generate: {e}")
            return {"question_text": f"Error generate soal.{e}", "grading_rubric": {}}

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
            response_content = await self._execute_with_retry(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile",
                response_format={"type": "json_object"}
            )
            
            return json.loads(response_content)
        except:
            print(f"ERROR Evaluate answer: {e}")
            return {"score": 0, "feedback": "Error menilai.", "is_correct": False}

    async def casual_chat(self, user_text: str, history: list = [], keyword_context: str = "", dataset_status: str = "NOT_FOUND"):
        
        if dataset_status == "FOUND":
            system_instruction = f"""
            [STATUS: VALID]
            User bertanya tentang topik teknis yang ADA dalam dataset skill kamu: **[{keyword_context}]**.
            
            TUGAS:
            1. Jawab pertanyaan user tentang topik tersebut dengan konseptual yang singkat, padat, dan mudah dimengerti.
            2. Fokus jawabanmu HANYA pada keyword tersebut.
            3. Gunakan analogi sederhana jika perlu. Jangan terlalu kaku seperti buku teks, tapi tetap akurat.
            4. Gaya bahasa: Ramah, Suportif, Mentor IT.
            5. Giring user untuk menggunakan fitur belajar seperti tanya tentang skill teknis, Ujian/Tes sub skill, cek progres, rekomendasi belajar.
            """
        else:
            # SKENARIO 2: Keyword Tidak Ditemukan (TOLAK)
            system_instruction = f"""
            [STATUS: INVALID / OUT OF SCOPE]
            User bertanya tentang topik yang TIDAK ditemukan dalam 'Skill Keywords Dataset' kamu.
            
            TUGAS:
            1. **TOLAK** untuk menjawab pertanyaan ini.
            2. Katakan dengan sopan seperti: "Maaf, topik ini tidak ada dalam database skill yang saya pelajari."
            3. JANGAN mencoba menjawab atau menebak, meskipun kamu tahu jawabannya secara umum. Patuhi whitelist dataset.
            5. Tawarkan user untuk menggunakan fitur belajar seperti tanya tentang skill teknis, Ujian/Tes sub skill, cek progres, rekomendasi belajar.
            """

        prompt_template = f"""
        [ROLE]
        Namamu MORA. Kamu adalah Mentor & Asisten Teknis Spesialis.
        
        {system_instruction}
            
        **BATASAN (BLACK LIST):**
            - Jika topik TIDAK ADA di skill keyword (misal: Mobil, Kendaraan, dll), TOLAK dengan sopan dan pivot ke materi silabus.
            - JANGAN BERIKAN KODE FULL. Jika user minta "Buatkan kodingan", arahkan mereka untuk mengambil Ujian/Tes. Kamu hanya menjelaskan *Konsep* dan *Logika*.
            - Hindari jawaban yang terlalu panjang (lebih dari 4 kalimat).
            - Jika tidak yakin, katakan "Maaf, itu di luar pengetahuan saya."
            - Jangan buat-buat jawaban untuk topik di luar silabus.

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
            return await self._execute_with_retry(
                messages=messages,
                model="llama-3.3-70b-versatile", 
                temperature=0.3
            )
        except Exception as e:
            print(f"ERROR Casual chat: {e}")
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
            return await self._execute_with_retry(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.1-8b-instant",
                temperature=0.7
            )
        except:
            print(f"ERROR Psych Analyze: {e}")
            return f"Kamu cocok jadi {role}!"

    async def analyze_progress(self, user_name: str, progress_data: dict):
        data_str = json.dumps(progress_data, indent=2)

        system_msg = {
            "role": "system", 
            "content": f"""
            Kamu adalah MORA, asisten belajar AI yang ceria, suportif, dan to-the-point.
            
            TUGAS:
            Analisis data progress student ini dan buat laporan singkat.
            
            DATA PROGRESS:
            {data_str}
            
            ATURAN FORMATTING (WAJIB MARKDOWN):
            1. Sapa user dengan namanya + emoji.
            2. Gunakan **Bold** untuk poin penting (nama course, skor, level).
            3. Pisahkan bagian menjadi dua kategori menggunakan Bullet Points:
               - 🏆 **Highlights** (Untuk course completed / naik level / skor tinggi).
               - 🚧 **Next Focus** (Untuk course yang masih in-progress/macet).
            4. Tutup dengan kalimat ajakan (Call to Action) yang semangat.
            5. Jangan terlalu panjang, maksimal 4-5 baris poin.
            
            Gaya Bahasa: Gaul, motivasi tinggi, pakai emoji (🚀, 🎉, 🔥).
            """
        }
        
        try:
            return await self._execute_with_retry(
                messages=[system_msg],
                model="llama-3.1-8b-instant", 
                temperature=0.7
            )
        except Exception as e:
            print(f"ERROR Analyze Progress: {e}")
            return f"Error generate progress: {str(e)}"

            
llm_engine = LLMEngine()