import pandas as pd 
import numpy as np
import os
# import matplotlib.pyplot as plt 
# import altair as alt

from rake_nltk import Rake
# from nltk.corpus import stopwords 
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from flask import Flask,render_template,request,jsonify,session

FILTERED_COURSES = None
SELECTED_COURSE = None

app = Flask(__name__)

class CourseData():
    a, b = None, None
	  

data = CourseData()


@app.route('/')
def index():
	df2 = load_data()
	# skills_avail = []
	# for i in range(1000):
	# 	skills_avail = skills_avail + df2['skills'][i]
	# skills_avail = list(set(skills_avail))
	skills_avail = load_skill()
	return render_template('index1.html',content=skills_avail)

@app.route("/ajax_add", methods=["POST","GET"])
def ajax_add():
	df5 = load_data()
	skills_avail = []
	for i in range(1000):
		skills_avail = skills_avail + df5['skills'][i]
	skills_avail = list(set(skills_avail))

	if request.method == "POST":
		# print("Inititally: ", request.form['recommend'])
		# print("Inititally: ", type(request.form['recommend']))

		#print("request.form: ", request.form.getlist('skills'))
		skills_selected = request.form.getlist('skills')

		#print("skills selected: ", skills_selected)
		#print("type of skills: ",type(skills_selected))
		temp = filter(df5, skills_selected, 'skills', 'course_url')

		#print("temp: ", temp)
		skill_filtered = df5[df5['course_url'].isin(temp)].reset_index()

		#print("skill_filtered: ", skill_filtered)
		courses = skill_filtered['course_name']

		data.a = courses
		

		courses_list = skill_filtered['course_name'].tolist()

		data.b = courses_list
		courses1 = skill_filtered[['course_name','course_url']]
		#print("courses: ", courses)
		#print("Number of courses selected: ", len(courses))

	# return render_template('index.html', data=courses.to_frame())
		# if request.form['recommend'] == "yes":
		# 	input_course = courses[0]
		# 	temp_sim, temp_dissim = content_based_recommendations(df5, input_course, courses)

		# 	return render_template('index312.html', tables=[temp_sim.to_html(classes='data'), temp_dissim.to_html(classes='data')], titles=['na','Similar Courses','Dissimilar Courses'], content=skills_selected)
		# else:
		return render_template('index.html',selected_skills=skills_selected, data=courses1, content=skills_avail, courses = courses_list, suggested=skill_filtered.to_dict(orient='records'))

@app.route("/ui_course_recommend", methods=["POST","GET"])
def ui_course_recommend():
	df6 = load_data()
	skills_avail = []
	for i in range(1000):
		skills_avail = skills_avail + df6['skills'][i]
	skills_avail = list(set(skills_avail))


	#print("Inititally: ", request.form['recommend'])
	#print("Inititally: ", type(request.form['recommend']))

	if request.form['recommend'] == "yes":
		input_course = request.form['course_selected']
		courses = data.a 
		skills_selected = data.b 
		temp_sim, temp_dissim = content_based_recommendations(df6, input_course, courses)

		return render_template('index312.html', sim=temp_sim.to_dict(orient='records'), content=skills_selected, selected_course = input_course)
	# else:
	# return render_template('index.html', data=courses1, content=skills_avail, courses = courses_list)


def clean_col_names(df, columns):
	new = []
	for c in columns:
		new.append(c.lower().replace(' ','_'))
	return new

def prepare_data(df):

	df.columns = clean_col_names(df, df.columns)

	
	df['skills'] = df['skills'].fillna('Missing')
	df['instructors'] = df['instructors'].fillna('Missing')

	
	def make_numeric(x):
		if(x=='Missing'):
			return np.nan
		return float(x)

	df['course_rating'] = df['course_rating'].apply(make_numeric)
	df['course_rated_by'] = df['course_rated_by'].apply(make_numeric)
	df['percentage_of_new_career_starts'] = df['percentage_of_new_career_starts'].apply(make_numeric)
	df['percentage_of_pay_increase_or_promotion'] = df['percentage_of_pay_increase_or_promotion'].apply(make_numeric)

	def make_count_numeric(x):
	    if('k' in x):
	        return (float(x.replace('k','')) * 1000)
	    elif('m' in x):
	        return (float(x.replace('m','')) * 1000000)
	    elif('Missing' in x):
	        return (np.nan)

	df['enrolled_student_count'] = df['enrolled_student_count'].apply(make_count_numeric)

	def find_time(x):
	    l = x.split(' ')
	    idx = 0
	    for i in range(len(l)):
	        if(l[i].isdigit()):
	            idx = i 
	    try:
	        return (l[idx] + ' ' + l[idx+1])
	    except:
	        return l[idx]

	df['estimated_time_to_complete'] = df['estimated_time_to_complete'].apply(find_time)

	
	def split_it(x):
		return (x.split(','))
	df['skills'] = df['skills'].apply(split_it)

	return df


def load_data():
	source_path1 = os.path.join("data/coursera-courses-overview.csv")
	source_path2 = os.path.join("data/coursera-individual-courses.csv")
	df_overview = pd.read_csv(source_path1)
	df_individual = pd.read_csv(source_path2)
	df = pd.concat([df_overview, df_individual], axis=1)


	df = prepare_data(df)

	return df
def load_skill():
	with open("./data/Skilllist.txt", "r") as file:
		data = eval(file.readline())
	return data

def filter(dataframe, chosen_options, feature, id):
	selected_records = []
	for i in range(1000):
		for op in chosen_options:
			if op in dataframe[feature][i]:
				selected_records.append(dataframe[id][i])
	return selected_records



def extract_keywords(df, feature):

    r = Rake()
    keyword_lists = []
    for i in range(df[feature].shape[0]):
        descr = df[feature][i]
        r.extract_keywords_from_text(descr)
        key_words_dict_scores = r.get_word_degrees()
        keywords_string = " ".join(list(key_words_dict_scores.keys()))
        keyword_lists.append(keywords_string)
        
    return keyword_lists

def recommendations(df, input_course, cosine_sim, find_similar=True, how_many=5):
    
    
    recommended = []
    #print("input_course: ", input_course)

    selected_course = df[df['course_name']==input_course]
    #print("input_course: ", type(input_course))

    #print("selected_course: ", selected_course)
    
    idx = selected_course.index[0]

    print("cosign similarity: ",cosine_sim)
    if(find_similar):
        score_series = pd.Series(cosine_sim[idx]).sort_values(ascending = False)
    else:
        score_series = pd.Series(cosine_sim[idx]).sort_values(ascending = True)
	
    print("score_series: ",score_series)
    if(len(score_series) < how_many):
    	how_many = len(score_series)
    top_sugg = list(score_series.iloc[1:how_many+1].index)
    
    print("top_sugg ",top_sugg)
    for i in top_sugg:
        qualified = df['course_name'].iloc[i]
        recommended.append(qualified)
        
    return recommended

def content_based_recommendations(df, input_course, courses):
	#print("input course: ", input_course)
	#print("courses: ", courses)
	
	# df = df[df['course_name'].isin(courses)].reset_index()
	#print("dataframe: ", df)

	#df = df[df['course_name'].isin(courses)].reset_index()


	#print("df.description: ", df["description"])

	df['skills_concat'] = df.apply(lambda x: " ".join(x['skills']),axis=1) 
	df['descr_keywords'] = extract_keywords(df, 'description')
	
	count = CountVectorizer()
	#print("df.description: ", df["description"])
	#count_matrix = count.fit_transform(df['descr_keywords'])
	count_matrix = count.fit_transform(df['skills_concat'])
	
	cosine_sim = cosine_similarity(count_matrix, count_matrix)

	
	rec_courses_similar = recommendations(df, input_course, cosine_sim, True)
	temp_sim = df[df['course_name'].isin(rec_courses_similar)]
	rec_courses_dissimilar = recommendations(df, input_course, cosine_sim, False)
	temp_dissim = df[df['course_name'].isin(rec_courses_dissimilar)]

	return temp_sim, temp_dissim

def prep_for_cbr(df):



	skills_avail = []
	for i in range(1000):
		skills_avail = skills_avail + df['skills'][i]
	skills_avail = list(set(skills_avail))
	
	skill_filtered = None
	courses = None
	input_course = "Nothing"
	
	temp = filter(df, skills_select, 'skills', 'course_url')
	skill_filtered = df[df['course_url'].isin(temp)].reset_index()
	
	courses = skill_filtered['course_name']
	


	
	
if __name__=="__main__":
	app.run(host="localhost", port=5000, debug=True)