import os
from flask import Flask, request, redirect, url_for,flash,render_template,send_from_directory
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import glob
import csv
# import os

# from pcd import *


def grayscale(source):
    row, col, ch = source.shape
    graykanvas = np.zeros((row, col, 1), np.uint8)
    for i in range(0, row):
        for j in range(0, col):
            blue, green, red = source[i, j]
            gray = red * 0.299 + green * 0.587 + blue * 0.114
            graykanvas.itemset((i, j, 0), gray)
    return graykanvas

def substract(img, subtractor):
    grey = grayscale(img)
    row, col, ch = img.shape
    canvas = np.zeros((row, col, 3), np.uint8)
    for i in range (0, row):
        for j in range(0, col):
            b, g, r = img[i,j]
            subs = int(grey[i,j]) - int(subtractor[i,j])
            if(subs<0):
                canvas.itemset((i, j, 0), 0)
                canvas.itemset((i, j, 1), 0)
                canvas.itemset((i, j, 2), 0)
            else:
                canvas.itemset((i, j, 0), b)
                canvas.itemset((i, j, 1), g)
                canvas.itemset((i, j, 2), r)
    return canvas


UPLOAD_FOLDER = 'images'
ALLOWED_EXTENSIONS = set([ 'png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/kematangan',  methods=['GET', 'POST'])
def utama():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            lokasi_file = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(lokasi_file)
            # utama(filename)

            filenamee = filename
            count = 1
            data = []
            
            img = cv2.imread(lokasi_file)
            # fix = tuple((300,300))
            # img = cv2.resize(img, fix)
            
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            gray = cv2.cvtColor(hsv, cv2.COLOR_RGB2GRAY)

            ret,biner_threshold = cv2.threshold(gray, 80, 255,cv2.THRESH_BINARY )

            kernel3 = np.ones((9, 9), np.uint8)
            dilation3 = cv2.dilate(biner_threshold, kernel3, iterations=15)
            erotion3 = cv2.erode(dilation3, kernel3, iterations=15)

            # cv2.imshow('gray', erotion3)
            # cv2.imshow('gray1', gray)

            biner_threshold = cv2.bitwise_not(erotion3)
            final = substract(img, biner_threshold)
            final1 = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)

            hitam = 0
            hijau = 0
            berat = 0
            full  = 0

            red = 0
            blue = 0
            green = 0

            r_size = 0
            b_size = 0
            g_size = 0

            #proposi RGB
            row, col = final1.shape
            for i in range(0, row):
                for j in range(0, col):
                    val = final1[i,j]
                    b, g, r = final[i,j]
                    #print(b,g,r)

                    #if(g!=0 and r!=0):
                    if (val!=0):
                        #if(b>20): hijau = hijau + 1
                        if(val>15 and val < 65): hijau=hijau+1
                        else: hitam = hitam+1

                    red = red + r
                    green = green + g
                    blue = blue + b

                    if(r): r_size = r_size + 1
                    if(g): g_size = g_size + 1
                    if(b): b_size = b_size + 1

            hijau_final = float(hijau)/(hitam+hijau)
            hitam_final = float(hitam)/(hitam+hijau)
            r_final = float(red)/r_size
            g_final = float(green)/g_size
            b_final = float(blue)/b_size


            berat = hitam+hijau
            full = row*col
            berat = float(berat)/full

            # return r_final, g_final, b_final, hijau_final, hitam_final, berat
            
            link=UPLOAD_FOLDER+filename
            # return render_template('index.html')
            # return render_template('index.html',filename=filename,value=r_final,value2=g_final,value3=b_final,value4=hijau_final,value5=hitam_final, value6=berat, file_url=link)
            #    r_final, g_final, b_final, hijau_final, hitam_final, berat
           
            #kodingan sisdas
            import pandas as pd
            mangga = pd.read_csv('mangga.csv', delimiter=';')

            """# Classification

            ## **Exploration Data**
            """

            #print(mangga.head())

            #print(mangga.describe().transpose())

            #print(mangga.shape)

            """## Train Test Split"""

            Xclass = mangga.drop('Kematangan',axis=1)
            Xclass = Xclass.drop('luas',axis=1)
            Xclass = Xclass.drop('Berat',axis=1)
            yclass = mangga['Kematangan']
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(Xclass, yclass)
            print()

            """## Preprocessing"""

            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            scaler.fit(X_train)
            StandardScaler(copy=True, with_mean=True, with_std=True)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)

            """## Training Model"""

            from sklearn.neural_network import MLPClassifier
            mlp = MLPClassifier(hidden_layer_sizes=(5,5,),max_iter=500)
            mlp.fit(X_train,y_train)

            """## Save Model"""
            import pickle
            # save the model to disk
            # filename = 'finalized_model.sav'
            # pickle.dump(mlp, open(filename, 'wb'))


            """## Prediction"""
            predictions = mlp.predict(X_test)
            from sklearn.metrics import classification_report,confusion_matrix
            #print(confusion_matrix(y_test,predictions))
            #print(classification_report(y_test,predictions))

            """## Buat ngetest klasifikasi"""
            #load model yang udah di save
            mlp = pickle.load(open('finalized_model.sav', 'rb'))
            #misal barisnya [[r_avg, g_avg, b_avg, hijau, hitam]]
            #testbaris = [[45.42237509758,46.6865241998439,13.7977100274688,0.966212919594067,0.0337870804059329]]
            testbaris = [[r_final, g_final, b_final, hijau_final, hitam_final]]
            #di praproses
            testbaris = scaler.transform(testbaris)
            #diprediksi
            predictions = mlp.predict(testbaris)       

            print(predictions)
            matang = predictions 
            if matang == [1]:
                matang = 'Kurang Matang'
            if matang == [2]:
                matang = 'Matang'
            if matang == [3]:
                matang = 'Sangat Matang'


            """## Buat ngetest regresi"""
            #misal input
            luas = berat
            #model
            berat = -54.6064 + 3184.4924*luas
            #truncate floating
            berat = '%.3f'%(berat)             
            return render_template('hasil.html',filename=filenamee,tingkat_matang=matang,berat=berat, file_url=link)
    
    return render_template('kematangan.html')




@app.route('/images/<filename>', methods=['GET'])
def show_file(filename):
    return send_from_directory('images/', filename, as_attachment=True)

@app.route('/')
def index():   
    return render_template('index.html',)


if __name__ == "__main__":
    app.run(debug=True)