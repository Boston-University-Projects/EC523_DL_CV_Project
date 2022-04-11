import flask
from flask import Flask, Response, request, render_template, redirect, url_for
from flaskext.mysql import MySQL
from collections import Counter
import os, base64

mysql = MySQL()
app = Flask(__name__)
app.secret_key = 'super secret string'  

app.config['MYSQL_DATABASE_USER'] = 'root'
app.config['MYSQL_DATABASE_PASSWORD'] = 'kotoric1025'
app.config['MYSQL_DATABASE_DB'] = 'cs523'
app.config['MYSQL_DATABASE_HOST'] = 'localhost'
mysql.init_app(app)

conn = mysql.connect()

#start upload photo
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/upload_photos', methods=['GET', 'POST'])
def upload_file():
	if request.method == 'POST':
		image= request.files['photo']
		data =image.read()
		id = request.form.get('id')
		cursor = conn.cursor()
		cursor.execute('''INSERT INTO Photos (data,photo_id) VALUES (%s, %s ) ''' ,( data,id))
		conn.commit()

		return render_template('result.html', photos=getPhotos(id),base64=base64)

	else:
		# print(" Arriving at GET upload method")
		return render_template('upload_photos.html')

def getPhotos(photo_id):
	cursor = conn.cursor()
	cursor.execute("SELECT data, photo_id FROM Photos WHERE photo_id = '{0}'".format( photo_id))
	return cursor.fetchall() #NOTE list of tuples, [(imgdata, pid), ...]


#default page
@app.route("/", methods=['GET'])
def hello():
	return render_template('homepage.html', message='Welecome')
	

if __name__ == "__main__":
	#this is invoked when in the shell  you run
	#$ python app.py
	app.run(port=5000, debug=True)
