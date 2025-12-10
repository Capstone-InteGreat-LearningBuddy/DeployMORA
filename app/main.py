from fastapi import FastAPI, HTTPException, Body
from app import schemas
from app.services.llm_engine import llm_engine
from app.services.skill_manager import skill_manager
import pandas as pd
import pickle
import ast
import os
from sklearn.metrics.pairwise import linear_kernel

app = FastAPI(title="MORA - AI Learning Assistant (Final)")

# --- GLOBAL MODELS STORE ---
models = {
    'df': None,
    'tfidf': None,
    'matrix': None
}

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
@app.post("/chat/process", response_model=schemas.ChatResponse)
async def process_chat(req: schemas.ChatRequest):
    # Ambil silabus skill berdasarkan role user untuk konteks AI
    role_data = skill_manager.get_role_data(req.role)
    skill_names = [s['name'] for s in role_data['sub_skills']] if role_data else []
    
    # Klasifikasi Niat User (Router LLM)
    intent = await llm_engine.process_user_intent(req.message, skill_names)
    
    action = intent.get('action')

    detected_skills = intent.get('detected_skills', []) 
    
    if action == "START_EXAM":
        target_skill_ids = []
        
        # 1. Cari ID untuk SEMUA skill yang dideteksi
        if detected_skills and role_data:
            for ds in detected_skills:
                for s in role_data['sub_skills']:
                    # Cek kemiripan nama
                    if s['name'].lower() in ds.lower() or ds.lower() in s['name'].lower():
                        if s['id'] not in target_skill_ids: # Cegah duplikat
                            target_skill_ids.append(s['id'])
        
        # 2. Generate Soal untuk SETIAP Skill ID yang ketemu
        if target_skill_ids:
            exam_questions_list = []
            
            for skid in target_skill_ids:
                # Ambil level user untuk skill ini
                user_current_level = req.current_skills.get(skid, "beginner")
                skill_details = skill_manager.get_skill_details(req.role, skid)
                level_data = skill_details['levels'].get(user_current_level, skill_details['levels']['beginner'])
                
                # Generate Soal via LLM (Tunggu satu-satu)
                llm_res = await llm_engine.generate_question(level_data['exam_topics'], user_current_level)
                
                # Masukkan ke list
                exam_questions_list.append({
                    "skill_id": skid,
                    "skill_name": skill_details['name'],
                    "level": user_current_level,
                    "question": llm_res['question_text'],
                    "context": llm_res['grading_rubric']
                })
            
            # 3. Kembalikan List Soal di dalam objek 'data'
            response_data = {
                "mode": "multiple_exams", # Penanda buat frontend
                "exams": exam_questions_list
            }
            
            # Buat kalimat sapaan dinamis
            skill_names_str = ", ".join([x['skill_name'] for x in exam_questions_list])
            final_reply = f"Siap! Saya menemukan {len(exam_questions_list)} topik: **{skill_names_str}**. Silakan kerjakan soal-soal berikut di bawah ini! 👇"

        else:
            # Fallback jika skill tidak dikenali
            action = "CASUAL_CHAT"
            final_reply = await llm_engine.casual_chat(req.message, [m.dict() for m in req.history])
    elif action == "GET_RECOMMENDATION":
        # Frontend yang harus lanjut memanggil endpoint /recommendations
        response_data = {"trigger_recommendation": True}
        final_reply = "Sedang menganalisis kebutuhan belajarmu..."

    elif action == "CASUAL_CHAT":
        final_reply = await llm_engine.casual_chat(req.message, [m.dict() for m in req.history])

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
@app.post("/progress")
def get_progress(req: schemas.ProgressRequest):
    role_data = skill_manager.get_role_data(req.role)
    if not role_data: return []
    
    progress_report = []
    level_weight = {"beginner": 0, "intermediate": 1, "advanced": 2}

    for skill in role_data['sub_skills']:
        skill_id = skill['id']
        user_level = req.current_skills.get(skill_id, "beginner")
        
        # Hitung Persen
        current_stage = level_weight.get(user_level, 0)
        percent = int((current_stage / 3) * 100)
        if user_level == "beginner": percent = 5
        elif user_level == "intermediate": percent = 50
        elif user_level == "advanced": percent = 80
        
        # Sisa tutorial (dummy/static logic karena detail ada di rekomendasi)
        remaining = 0 
        
        progress_report.append({
            "skill_name": skill['name'],
            "current_level": user_level,
            "progress_percent": percent,
            "remaining_tutorials": remaining
        })
        
    return progress_report