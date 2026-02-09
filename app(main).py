import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt

# --- Data and Model Setup ---
# Replace the path below with your actual dataset path
DATA_PATH = "d:\\new folder\\assignment\\ml lab\\lab project\\stud.csv"
MODEL_PATH = "model.pkl"

# Load and preprocess dataset
data = pd.read_csv(DATA_PATH)
data['Courses_label'] = LabelEncoder().fit_transform(data['Courses'])

categorical_columns = ['Drawing','Dancing','Singing','Sports','Video Game','Acting','Travelling','Gardening',
                       'Animals','Photography','Teaching','Exercise','Coding','Electricity Components',
                       'Mechanic Parts','Computer Parts','Researching','Architecture','Historic Collection',
                       'Botany','Zoology','Physics','Accounting','Economics','Sociology','Geography',
                       'Psycology','History','Science','Bussiness Education','Chemistry','Mathematics',
                       'Biology','Makeup','Designing','Content writing','Crafting','Literature','Reading',
                       'Cartooning','Debating','Asrtology','Hindi','French','English','Urdu','Other Language',
                       'Solving Puzzles','Gymnastics','Yoga','Engineering','Doctor','Pharmisist','Cycling',
                       'Knitting','Director','Journalism','Bussiness','Listening Music']

for col in categorical_columns:
    data[col] = LabelEncoder().fit_transform(data[col])

X = data[categorical_columns]
Y = data['Courses_label']

# Load or train model
try:
    model = joblib.load(MODEL_PATH)
except:
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_PATH)

numeric_to_category = { 
    0: 'Animation, Graphics and Multimedia',
    1: 'B.Arch- Bachelor of Architecture',
    2: 'B.Com- Bachelor of Commerce',
    3: 'B.Ed.',
    4: 'B.Sc- Applied Geology',
    5: 'B.Sc- Nursing',
    6: 'B.Sc. Chemistry',
    7: 'B.Sc. Mathematics',
    8: 'B.Sc.- Information Technology',
    9: 'B.Sc.- Physics',
    10: 'B.Tech.-Civil Engineering',
    11: 'B.Tech.-Computer Science and Engineering',
    12: 'B.Tech.-Electrical and Electronics Engineering',
    13: 'B.Tech.-Electronics and Communication Engineering',
    14: 'B.Tech.-Mechanical Engineering',
    15: 'BA in Economics',
    16: 'BA in English',
    17: 'BA in Hindi',
    18: 'BA in History',
    19: 'BBA- Bachelor of Business Administration',
    20: 'BBS- Bachelor of Business Studies',
    21: 'BCA- Bachelor of Computer Applications',
    22: 'BDS- Bachelor of Dental Surgery',
    23: 'BEM- Bachelor of Event Management',
    24: 'BFD- Bachelor of Fashion Designing',
    25: 'BJMC- Bachelor of Journalism and Mass Communication',
    26: 'BPharma- Bachelor of Pharmacy',
    27: 'BTTM- Bachelor of Travel and Tourism Management',
    28: 'BVA- Bachelor of Visual Arts',
    29: 'CA- Chartered Accountancy',
    30: 'CS- Company Secretary',
    31: 'Civil Services',
    32: 'Diploma in Dramatic Arts',
    33: 'Integrated Law Course- BA + LL.B',
    34: 'MBBS'
}

# Holland Personality Test Data
holland_questions = {
    "R": ["I enjoy working with machines and tools.", "I like to work with numbers and solve mathematical problems.", "I prefer practical tasks over abstract ones."],
    "I": ["I enjoy solving puzzles and brain teasers.", "I like conducting experiments and exploring new ideas.", "I enjoy analyzing data to find patterns and trends."],
    "A": ["I enjoy drawing, painting, or creating visual art.", "I like expressing myself through music or dance.", "I like writing poetry or stories."],
    "S": ["I enjoy helping people solve their problems.", "I like volunteering and contributing to my community.", "I enjoy teaching and educating others."],
    "E": ["I enjoy taking on leadership roles and responsibilities.", "I like persuading and convincing others.", "I like organizing events and gatherings."],
    "C": ["I prefer working with numbers and data.", "I like creating and following organized systems.", "I enjoy record-keeping and data analysis."]
}

personality_info = {
    "R": {"name": "Realistic", "description": "Practical, hands-on, enjoy tools/machines.", "careers": ["Carpenter","Electrician","Mechanic","Plumber","Welder"]},
    "I": {"name": "Investigative", "description": "Analytical, enjoy problem-solving.", "careers": ["Scientist","Engineer","Researcher","Programmer","Mathematician"]},
    "A": {"name": "Artistic", "description": "Creative and expressive.", "careers": ["Artist","Graphic Designer","Writer","Interior Designer","Photographer"]},
    "S": {"name": "Social", "description": "Compassionate, enjoy helping others.", "careers": ["Teacher","Social Worker","Nurse","Counselor","Psychologist"]},
    "E": {"name": "Enterprising", "description": "Ambitious, enjoy leadership.", "careers": ["Entrepreneur","Sales Manager","Marketing Manager","Consultant","Politician"]},
    "C": {"name": "Conventional", "description": "Detail-oriented, organized.", "careers": ["Accountant","Financial Analyst","Data Analyst","Office Manager","Banker"]}
}

# Sample government colleges for dashboard
government_colleges = [
    {'name':'Govt College A','distance':5,'courses':['Science','Engineering'],'fees':20000},
    {'name':'Govt College B','distance':12,'courses':['Commerce','Business'],'fees':15000},
    {'name':'Govt College C','distance':7,'courses':['Arts','Social Sciences'],'fees':10000},
    {'name':'Govt College D','distance':10,'courses':['Vocational','Technical'],'fees':8000},
]

scholarship_schemes = {
    'Science': ['Govt Scheme 1: Apply by Dec 31', 'Science Excellence Scholarship'],
    'Commerce': ['Commerce Merit Scholarship', 'Govt Scheme 2: Apply by Nov 30'],
    'Arts': ['Arts Talent Scholarship', 'Govt Scheme 3: Open all year'],
    'Vocational': ['Skill Development Grant', 'Vocational Training Scholarship']
}

stream_map = {
    'Realistic': 'Science',
    'Investigative': 'Science',
    'Artistic': 'Arts',
    'Social': 'Arts',
    'Enterprising': 'Commerce',
    'Conventional': 'Commerce'
}

# --- App Class ---

class CareerRecommenderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Integrated Career advisor System")

        self.aptitude_vars = {feat: tk.IntVar() for feat in categorical_columns}
        self.personality_scores = {ptype:0 for ptype in holland_questions.keys()}
        self.personality_index = 0
        self.personality_types = list(holland_questions.keys())
        self.user_profile = {}

        self.show_aptitude_screen()

    def show_aptitude_screen(self):
        self.clear_root()
        frame = ttk.Frame(self.root)
        frame.pack(padx=10, pady=10, fill='both', expand=True)
        ttk.Label(frame, text="Select your interests/aptitudes (check all that apply):").pack()

        check_frame = ttk.Frame(frame)
        check_frame.pack(pady=5)
        cols = 3
        for i, feat in enumerate(categorical_columns):
            cb = ttk.Checkbutton(check_frame, text=feat, variable=self.aptitude_vars[feat])
            cb.grid(row=i//cols, column=i%cols, sticky='w', padx=5, pady=2)

        ttk.Button(frame, text="Proceed to Personality Test", command=self.start_personality_test).pack(pady=10)

    def start_personality_test(self):
        self.user_profile['aptitude'] = {feat: var.get() for feat, var in self.aptitude_vars.items()}
        self.personality_index = 0
        self.personality_scores = {ptype:0 for ptype in self.personality_types}
        self.show_next_personality_set()

    def show_next_personality_set(self):
        if self.personality_index >= len(self.personality_types):
            self.compute_recommendations()
            return

        ptype = self.personality_types[self.personality_index]
        questions = holland_questions[ptype]

        self.personality_window = tk.Toplevel(self.root)
        self.personality_window.title(f"Holland Personality Test - {personality_info[ptype]['name']}")

        self.answers_vars = []
        for q in questions:
            ttk.Label(self.personality_window, text=q).pack(anchor='w', padx=10, pady=3)
            var = tk.IntVar()
            self.answers_vars.append(var)
            for i, opt in enumerate(["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"]):
                ttk.Radiobutton(self.personality_window, text=opt, variable=var, value=i).pack(anchor='w', padx=20)

        ttk.Button(self.personality_window, text="Submit", command=self.submit_personality_answers).pack(pady=10)

    def submit_personality_answers(self):
        score = sum(var.get() for var in self.answers_vars)
        ptype = self.personality_types[self.personality_index]
        self.personality_scores[ptype] = score

        self.personality_index += 1
        self.personality_window.destroy()
        self.show_next_personality_set()

    def compute_recommendations(self):
        user_aptitude = pd.DataFrame([self.user_profile['aptitude']])
        for col in X.columns:
            if col not in user_aptitude.columns:
                user_aptitude[col] = 0

        prediction_num = model.predict(user_aptitude)[0]
        predicted_course = numeric_to_category.get(prediction_num, "Unknown Course")
        self.user_profile['predicted_course'] = predicted_course

        dominant_personality = max(self.personality_scores, key=self.personality_scores.get)
        personality_summary = personality_info[dominant_personality]
        self.user_profile['personality'] = personality_summary

        user_stream = stream_map.get(personality_summary['name'], 'Vocational')
        self.user_profile['recommended_stream'] = user_stream

        suitable_colleges = [c for c in government_colleges
                            if user_stream in c['courses'] and c['distance'] <= 15]
        self.user_profile['selected_colleges'] = suitable_colleges

        scholarships = scholarship_schemes.get(user_stream, [])
        self.user_profile['scholarship_alerts'] = scholarships

        self.show_dashboard()

    def show_dashboard(self):
        self.clear_root()
        frame = ttk.Frame(self.root)
        frame.pack(padx=10, pady=10, fill='both', expand=True)

        ttk.Label(frame, text="=== Career Recommendation Dashboard ===").pack(pady=5)
        ttk.Label(frame, text=f"ML Predicted Course: {self.user_profile['predicted_course']}").pack(anchor='w', padx=5)
        ttk.Label(frame, text=f"Holland Personality Type: {self.user_profile['personality']['name']}").pack(anchor='w', padx=5)
        ttk.Label(frame, text=f"Personality Description: {self.user_profile['personality']['description']}").pack(anchor='w', padx=5)

        ttk.Label(frame, text="Personality-Based Career Suggestions:").pack(anchor='w', padx=5)
        for career in self.user_profile['personality']['careers']:
            ttk.Label(frame, text=f" - {career}").pack(anchor='w', padx=20)

        ttk.Label(frame, text="Suitable Nearby Government Colleges (≤15km):").pack(anchor='w', padx=5, pady=(10,0))
        for college in self.user_profile['selected_colleges']:
            ttk.Label(frame, text=f" - {college['name']} (Distance: {college['distance']}km, Fees: ₹{college['fees']})").pack(anchor='w', padx=20)

        ttk.Label(frame, text="Scholarship Alerts:").pack(anchor='w', padx=5, pady=(10,0))
        for alert in self.user_profile['scholarship_alerts']:
            ttk.Label(frame, text=f" - {alert}").pack(anchor='w', padx=20)

        ttk.Button(frame, text="Show Personality Distribution Chart", command=self.show_donut_chart).pack(pady=10)
        ttk.Button(frame, text="Restart", command=self.show_aptitude_screen).pack(pady=5)

    def show_donut_chart(self):
        labels = [personality_info[ptype]['name'] for ptype in self.personality_scores.keys()]
        scores = list(self.personality_scores.values())
        total = sum(scores)
        percentages = [(score/total)*100 if total > 0 else 0 for score in scores]

        fig, ax = plt.subplots()
        ax.pie(percentages, labels=labels, autopct='%1.1f%%', startangle=90, pctdistance=0.85)
        center_circle = plt.Circle((0,0), 0.70, fc='white')
        fig.gca().add_artist(center_circle)
        ax.axis('equal')
        plt.title("Personality Type Distribution")
        plt.show()

    def clear_root(self):
        for widget in self.root.winfo_children():
            widget.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = CareerRecommenderApp(root)
    root.mainloop()