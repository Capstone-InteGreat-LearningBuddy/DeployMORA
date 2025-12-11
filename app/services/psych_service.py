# app/services/psych_service.py
from typing import Dict

# Bank Soal Psikologis (Bisa ditambah nanti)
PSYCH_QUESTIONS = [
    {
        "id": 1,
        "question": "Mana kegiatan yang paling relate denganmu di pagi hari?",
        "options": {
            "A": "Baca atau lihat info viral dari berbagai sumber",
            "B": "Coret-coret atau menulis di buku"
        },
        "role_mapping": {"A": "AI Engineer", "B": "Front-End Web Developer"}
    },
    {
        "id": 2,
        "question": "JJika sedang menghadapi masalah, cara mana yang paling mirip denganmu?",
        "options": {
            "A": "Cari tahu masalahnya dari berbagai macam sudut pandang",
            "B": "Ngobrol dengan teman untuk mendapatkan ide baru"
        },
        "role_mapping": {"A": "AI Engineer", "B": "Front-End Web Developer"}
    },
    {
        "id": 3,
        "question": "Kalau lagi bermain sosial media, mana aktivitas yang paling relate denganmu?",
        "options": {
            "A": "Stalking akun-akun yang suka share fakta-fakta seru",
            "B": "Share-share postingan teman sambil comment"
        },
        "role_mapping": {"A": "AI Engineer", "B": "Front-End Web Developer"}
    },
    {
        "id": 4,
        "question": "Nah, kalau lagi liburan, kegiatan mana yang paling bikin kamu excited?",
        "options": {
            "A": "Mencari review tempat yang akan dikunjungi",
            "B": "Membuat konten blog atau vlog tentang petualangan liburan"
        },
        "role_mapping": {"A": "AI Engineer", "B": "Front-End Web Developer"}
    },
    {
        "id": 5,
        "question": "Kalau lagi kerja bareng tim, mana peran yang paling mirip dengamu?",
        "options": {
            "A": "Jadi orang yang bantu tim ngambil keputusan dengan analisis situasi",
            "B": "Jadi orang yang bikin presentasi buat menyampaikan ide-ide tim"
        },
        "role_mapping": {"A": "AI Engineer", "B": "Front-End Web Developer"}
    }
]

class PsychService:
    def get_all_questions(self):
        """Mengembalikan soal tanpa kunci jawaban (untuk frontend)"""
        return [
            {
                "id": q["id"],
                "question": q["question"],
                "options": q["options"]
            }
            for q in PSYCH_QUESTIONS
        ]

    def calculate_result(self, user_answers: Dict[int, str]):
        """
        Menghitung skor berdasarkan jawaban user.
        user_answers contoh: {1: "A", 2: "B"}
        """
        scores = {"AI Engineer": 0, "Front-End Web Developer": 0}
        
        # Simpan trait kepribadian user untuk dikirim ke LLM nanti
        user_traits = []

        for q in PSYCH_QUESTIONS:
            q_id = q["id"]
            user_choice = user_answers.get(q_id) # "A" atau "B"
            
            if user_choice and user_choice in q["role_mapping"]:
                # 1. Tambah Skor
                role = q["role_mapping"][user_choice]
                scores[role] += 1
                
                # 2. Catat trait (pilihan user) untuk konteks LLM
                chosen_text = q["options"][user_choice]
                user_traits.append(f"- Lebih suka: {chosen_text}")

        # Tentukan Pemenang
        winner_role = max(scores, key=scores.get)
        
        return {
            "winner": winner_role,
            "scores": scores,
            "traits": user_traits
        }

psych_service = PsychService()