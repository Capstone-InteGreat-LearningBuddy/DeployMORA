from fastapi import FastAPI, HTTPException, Body
from app import schemas
from app.services.llm_engine import llm_engine
from app.services.skill_manager import skill_manager
import pandas as pd
import pickle
import ast
import os
from sklearn.metrics.pairwise import linear_kernel
from app.services.psych_service import psych_service
from typing import List

app = FastAPI(title="MORA - AI Learning Assistant (Final)")

# --- GLOBAL MODELS STORE ---
models = {
    'df': None,
    'tfidf': None,
    'matrix': None
}

SKILL_KEYWORDS = []

@app.on_event("startup")
def load_skill_keywords():
    global SKILL_KEYWORDS
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(current_dir, "data", "Skill Keywords.csv")
        df = pd.read_csv(csv_path)
        SKILL_KEYWORDS = df['keyword'].dropna().tolist()
        print(f"✅ Berhasil memuat {len(SKILL_KEYWORDS)} keywords skill.")
    except Exception as e:
        print(f"⚠️ Gagal memuat dataset keyword: {e}")
        SKILL_KEYWORDS = []

# Fungsi Pembantu: Mencari keyword dalam pesan user
def find_keywords_in_text(user_text: str):
    found = []
    text_lower = " " + user_text.lower() + " " # Tambah spasi biar aman deteksi kata pendek
    
    for k in SKILL_KEYWORDS:
        # Cek sederhana: Apakah keyword ada di dalam pesan?
        # Untuk kata pendek (<3 huruf) seperti "C", "R", "Go", kita pakai spasi agar tidak match "Car" atau "Goat"
        if len(k) < 3:
            if f" {k.lower()} " in text_lower:
                found.append(k)
        else:
            if k.lower() in text_lower:
                found.append(k)
                
    # Hapus duplikat dan kembalikan
    return list(set(found))

# --- 1. STARTUP: LOAD MODEL .PKL ---
@app.on_event("startup")
def load_models():
    print("🔄 Loading Pre-trained Models...")
    
    # Menggunakan Absolute Path agar aman dijalankan dari mana saja
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(current_dir)
    artifacts_dir = os.path.join(base_dir, "model_artifacts")
    
    try:
        with open(os.path.join(artifacts_dir, 'courses_df.pkl'), 'rb') as f:
            models['df'] = pickle.load(f)
        with open(os.path.join(artifacts_dir, 'tfidf_vectorizer.pkl'), 'rb') as f:
            models['tfidf'] = pickle.load(f)
        with open(os.path.join(artifacts_dir, 'tfidf_matrix.pkl'), 'rb') as f:
            models['matrix'] = pickle.load(f)
        print(f"✅ Models Loaded Successfully from: {artifacts_dir}")
    except Exception as e:
        print(f"❌ Error Loading Models: {e}")
        print(f"👉 Pastikan folder 'model_artifacts' ada di: {base_dir}")

# --- 2. ENDPOINT REKOMENDASI (ML POWERED) ---
@app.post("/recommendations")
def get_recommendations(user: schemas.UserProfile):
    df = models.get('df')
    tfidf = models.get('tfidf')
    matrix = models.get('matrix')
    
    # Jika model belum siap, return kosong biar gak crash
    if df is None: return []

    # Mapping Level agar komputer mengerti urutan
    LEVEL_MAP = {
        'beginner': 1, 'dasar': 1, 'pemula': 1,
        'intermediate': 2, 'menengah': 2,
        'advanced': 3, 'mahir': 3, 'expert': 3, 'profesional': 3
    }
    
    final_recs = []
    # Set course yang sudah diambil agar tidak disarankan lagi
    seen_courses = set(user.completed_courses)
    
    # --- LOGIKA CORE: Loop setiap 'Gap' Skill User ---
    for gap in user.missing_skills:
        skill_query = gap.skill_name
        target_lvl_str = gap.target_level.lower()
        target_lvl_num = LEVEL_MAP.get(target_lvl_str, 1) # Default 1 (Pemula)
        
        try:
            # 1. Transform nama skill jadi vektor angka
            vec = tfidf.transform([skill_query.lower()])
            
            # 2. Hitung kemiripan (Cosine Similarity)
            scores = linear_kernel(vec, matrix).flatten()
            
            # 3. Ambil Top 15 kandidat
            indices = scores.argsort()[:-15:-1]
            
            for idx in indices:
                score = scores[idx]
                # Filter awal: Skip jika kemiripan text terlalu rendah
                if score < 0.1: continue
                
                course = df.iloc[idx]
                c_id = int(course['course_id'])
                
                if c_id in seen_courses: continue
                
                # --- FILTER LEVEL (ADAPTIVE) ---
                c_lvl_str = str(course['level_name']).lower()
                c_lvl_num = LEVEL_MAP.get(c_lvl_str, 1)
                
                # Logic: Jangan kasih course yang levelnya DI ATAS target (kejauhan)
                if c_lvl_num > target_lvl_num: continue
                
                # Logic Badge (Penanda)
                if c_lvl_num == target_lvl_num:
                    badge = "🎯 Target Pas"
                else:
                    badge = "↺ Review Dasar"

                # Parse Tutorial List (karena di CSV formatnya string)
                tuts = course['tutorial_list']
                if isinstance(tuts, str):
                    try: tuts = ast.literal_eval(tuts)
                    except: tuts = []
                
                # Tambahkan ke hasil
                final_recs.append({
                    "skill": skill_query,
                    "current_level": gap.target_level,
                    "course_to_take": course['course_name'],
                    "chapters": tuts[:3], # Ambil 3 bab pertama
                    "match_score": round(score * 100, 1),
                    "badge": badge
                })
                seen_courses.add(c_id)
                
        except Exception as e:
            print(f"Error processing {skill_query}: {e}")
            continue

    # Urutkan berdasarkan skor kecocokan tertinggi
    final_recs = sorted(final_recs, key=lambda x: x['match_score'], reverse=True)
    
    return final_recs[:5] # Kembalikan Top 5

# --- 3. ENDPOINT CHAT ROUTER ---
# app/main.py (Bagian process_chat saja)

@app.post("/chat/process", response_model=schemas.ChatResponse)
async def process_chat(req: schemas.ChatRequest):
    role_data = skill_manager.get_role_data(req.role)
    # --- [UPDATE BARU: Ektrak Silabus Lengkap] ---
    # Kita buat string rapi berisi Skill + Topik-topiknya
    found_keywords = find_keywords_in_text(req.message)
    
    # Siapkan context string untuk dikirim ke LLM
    if found_keywords:
        # Jika ketemu: "User bertanya tentang: Python, SQL"
        keyword_context = ", ".join(found_keywords)
        dataset_status = "FOUND"
    else:
        # Jika tidak ketemu
        keyword_context = "NONE"
        dataset_status = "NOT_FOUND"

    # 2. Router
    intent = await llm_engine.process_user_intent(req.message, [])
    
    action = intent.get('action')
    # PERUBAHAN 1: Ambil List skills, bukan single skill
    detected_skills_list = intent.get('detected_skills', [])
    
    final_reply = ""
    response_data = None
    
    # 3. Logic
    if action == "START_EXAM":
        target_skill_ids = []
        
        # A. Cari ID untuk SEMUA skill yang dideteksi (Looping)
        if detected_skills_list and role_data:
            for ds in detected_skills_list:
                for s in role_data['sub_skills']:
                    # Cek kemiripan nama
                    if s['name'].lower() in ds.lower() or ds.lower() in s['name'].lower():
                        if s['id'] not in target_skill_ids:
                            target_skill_ids.append(s['id'])
        
        # B. Jika ada skill yang valid, generate soal untuk MASING-MASING skill
        if target_skill_ids:
            exam_list = []
            
            for skid in target_skill_ids:
                # Ambil level user
                user_current_level = req.current_skills.get(skid, "beginner")
                skill_details = skill_manager.get_skill_details(req.role, skid)
                level_data = skill_details['levels'].get(user_current_level, skill_details['levels']['beginner'])
                
                # Generate Soal (Sequential)
                llm_res = await llm_engine.generate_question(level_data['exam_topics'], user_current_level)
                
                # Masukkan ke list soal
                exam_list.append({
                    "skill_id": skid,
                    "skill_name": skill_details['name'],
                    "level": user_current_level,
                    "question": llm_res['question_text'],
                    "context": llm_res['grading_rubric']
                })
            
            # C. Format Response Baru (Multi-Exam)
            response_data = {
                "mode": "multiple_exams", # Penanda buat frontend
                "exams": exam_list        # List soal ada di sini
            }
            
            skill_display = ", ".join([x['skill_name'] for x in exam_list])
            final_reply = f"Siap! Saya siapkan {len(exam_list)} ujian untukmu: **{skill_display}**. Silakan kerjakan satu per satu di bawah ini! 👇"
            
        else:
            action = "CASUAL_CHAT"
            final_reply = await llm_engine.casual_chat(
            req.message, 
            [m.dict() for m in req.history], 
            keyword_context, 
            dataset_status 
        )

    elif action == "START_PSYCH_TEST":
        response_data = {"trigger_psych_test": True}
        final_reply = "Tenang, Mora punya tes kepribadian singkat untuk membantumu memilih job role antara **AI Engineer** atau **Front-End Developer**. Yuk coba sekarang! 👇"

    elif action == "GET_RECOMMENDATION":
        response_data = {"trigger_recommendation": True}
        final_reply = "Sedang menganalisis kebutuhan belajarmu..."

    elif action == "CASUAL_CHAT":
        final_reply = await llm_engine.casual_chat(
            req.message, 
            [m.dict() for m in req.history], 
            keyword_context, 
            dataset_status 
        )

    return schemas.ChatResponse(
        reply=final_reply,
        action_type=action,
        data=response_data
    )


@app.post("/exam/submit", response_model=schemas.EvaluationResponse)
async def submit_exam(sub: schemas.AnswerSubmission):
    evaluation = await llm_engine.evaluate_answer(
        user_answer=sub.user_answer,
        question_context={
            "question_text": "REFER TO CONTEXT",
            "grading_rubric": sub.question_context
        }
    )
    
    is_passed = evaluation['is_correct'] and evaluation['score'] >= 70
    suggested_lvl = "intermediate" if is_passed else None # Logika sederhana
    
    return schemas.EvaluationResponse(
        is_correct=evaluation['is_correct'],
        score=evaluation['score'],
        feedback=evaluation['feedback'],
        passed=is_passed,
        suggested_new_level=suggested_lvl
    )

# --- 5. ENDPOINT PROGRESS ---
@app.post("/progress/analyze")
async def get_progress_analysis(data: schemas.ProgressData):
    # Konversi objek Pydantic ke Dictionary biasa
    progress_dict = data.dict()
    
    # Panggil LLM khusus analisis
    analysis_text = await llm_engine.analyze_progress(
        user_name=data.user_name, 
        progress_data=progress_dict
    )
    
    return {"analysis": analysis_text}

# ==========================================
# ENDPOINT PSIKOLOGI (JOB ROLE TEST)
# ==========================================

@app.get("/psych/questions", response_model=List[schemas.PsychQuestionItem])
def get_psych_questions():
    """Mengambil daftar soal tes kepribadian."""
    return psych_service.get_all_questions()

@app.post("/psych/submit", response_model=schemas.PsychResultResponse)
async def submit_psych_test(req: schemas.PsychSubmitRequest):
    """Menerima jawaban user, hitung skor, dan minta analisis LLM."""
    
    # 1. Hitung Skor secara matematis
    result = psych_service.calculate_result(req.answers)
    
    winner = result["winner"]
    scores = result["scores"]
    traits = result["traits"]
    
    # 2. Minta LLM buatkan kata-kata mutiara/analisis
    analysis_text = await llm_engine.analyze_psych_result(winner, traits)
    
    return schemas.PsychResultResponse(
        suggested_role=winner,
        analysis=analysis_text,
        scores=scores
    )