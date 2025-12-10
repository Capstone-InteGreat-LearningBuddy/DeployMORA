# app/services/psych_service.py

# Bank Soal Psikologis (Bisa ditambah nanti)
PSYCH_QUESTIONS = [
    {
        "id": 1,
        "question": "Mana kegiatan yang paling relate denganmu di pagi hari?",
        "options": {
            "A": "Baca atau lihat info viral dari berbagai sumber (Cari Pola/Data)",
            "B": "Coret-coret ide atau menulis jurnal di buku (Visual/Desain)"
        },
        "role_mapping": {"A": "AI Engineer", "B": "Front-End Web Developer"}
    },
    {
        "id": 2,
        "question": "Jika kamu melihat sebuah website yang jelek, apa yang pertama kali kamu pikirkan?",
        "options": {
            "A": "Ini fitur search-nya lambat banget, pasti database-nya berantakan.",
            "B": "Ini warnanya nabrak banget, font-nya juga susah dibaca."
        },
        "role_mapping": {"A": "AI Engineer", "B": "Front-End Web Developer"}
    },
    {
        "id": 3,
        "question": "Saat memecahkan masalah, gaya kamu lebih seperti apa?",
        "options": {
            "A": "Mengumpulkan banyak data dulu, baru menyimpulkan solusi.",
            "B": "Mencoba menggambar sketsa solusi dulu, baru diperbaiki sambil jalan."
        },
        "role_mapping": {"A": "AI Engineer", "B": "Front-End Web Developer"}
    }
]

def get_psych_question(index: int):
    if index < len(PSYCH_QUESTIONS):
        return PSYCH_QUESTIONS[index]
    return None