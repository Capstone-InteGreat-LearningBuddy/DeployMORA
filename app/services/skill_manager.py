import json
import os

# Lokasi file JSON
JSON_PATH = os.path.join(os.path.dirname(__file__), "../data/Sub_skill.json")

class SkillManager:
    def __init__(self):
        self.data = self._load_data()

    def _load_data(self):
        with open(JSON_PATH, 'r') as f:
            return json.load(f)

    def get_role_data(self, role_name: str):
        """Mengambil data skill berdasarkan role (AI Engineer / Web Dev)"""
        for role in self.data:
            if role['role_name'].lower() == role_name.lower():
                return role
        return None

    def get_skill_details(self, role_name: str, skill_id: str):
        """Mengambil detail satu skill spesifik"""
        role_data = self.get_role_data(role_name)
        if not role_data:
            return None
        
        for skill in role_data['sub_skills']:
            if skill['id'] == skill_id:
                return skill
        return None

# Instance global
skill_manager = SkillManager()