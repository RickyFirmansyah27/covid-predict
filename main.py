import streamlit as st
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from jcopml.pipeline import cat_pipe, num_pipe
from sklearn.naive_bayes import GaussianNB
from jcopml.plot import plot_confusion_matrix
from plotly.subplots import make_subplots
from sklearn import tree


def DataTrainingTree():
    X = df.drop(columns="Indikasi")
    y = df.Indikasi
    X_train, X_test , y_train, y_test = train_test_split(X,y, test_size=0.2 ,random_state=42)
    preprocessor = ColumnTransformer([('numeric', num_pipe(),["Temperatur"]),("categoric", cat_pipe(encoder='onehot'),["Tenggorokan", "Badan", "Paru-Paru",]),])
    pipeline = Pipeline([('prep', preprocessor),('algo',tree.DecisionTreeClassifier())])
    pipeline.fit(X_train,y_train)
    plot_confusion_matrix(X_train,y_train,X_test,y_test, pipeline)
  
    

def DataTestingTree():
    DataTrainingTree()
    X = df.drop(columns="Indikasi")
    y = df.Indikasi
    X_train, X_test , y_train, y_test = train_test_split(X,y, test_size=0.2 ,random_state=42)
    preprocessor = ColumnTransformer([('numeric', num_pipe(),["Temperatur"]),
    ("categoric", cat_pipe(encoder='onehot'),["Tenggorokan", "Badan", "Paru-Paru",]),])
    pipeline = Pipeline([('prep', preprocessor),('algo',tree.DecisionTreeClassifier())])
    pipeline.fit(X_train,y_train)
    X_pred['Indikasi'] = pipeline.predict(X_pred)
    display_results(X_train,X_pred)
    ShowTree()

def prediksi_tree(tenggorokan,temperatur,badan,paru):
    X = df.drop(columns="Indikasi")
    y = df.Indikasi
    X_train, X_test , y_train, y_test = train_test_split(X,y, test_size=0.2 ,random_state=42)
    df_tree = pd.DataFrame(
     {'Tenggorokan' : [tenggorokan],
      'Temperatur'  : [temperatur],
      'Badan'       : [badan],
      'Paru-Paru'   : [paru]
     }
    )
    preprocessor = ColumnTransformer([('numeric', num_pipe(),["Temperatur"]),
    ("categoric", cat_pipe(encoder='onehot'),["Tenggorokan", "Badan", "Paru-Paru",]),])
    pipeline = Pipeline([('prep', preprocessor),('algo',tree.DecisionTreeClassifier())])
    testing = df_tree
    pipeline.fit(X_train,y_train)
    testing['Indikasi'] = pipeline.predict(testing)
    st.write(testing)
    

def DataTraining():
    X = df.drop(columns="Indikasi")
    y = df.Indikasi
    X_train, X_test , y_train, y_test = train_test_split(X,y, test_size=0.2 ,random_state=42)
    preprocessor = ColumnTransformer([('numeric', num_pipe(),["Temperatur"]),("categoric", cat_pipe(encoder='onehot'),["Tenggorokan", "Badan", "Paru-Paru",]),])
    pipeline = Pipeline([('prep', preprocessor),('algo',GaussianNB())])
    pipeline.fit(X_train,y_train)
    plot_confusion_matrix(X_train,y_train,X_test,y_test, pipeline)

def ShowNB():
    st.write('')
    st.subheader('Plot Confussion Matrix')
    X = df.drop(columns="Indikasi")
    y = df.Indikasi
    X_train, X_test , y_train, y_test = train_test_split(X,y, test_size=0.2 ,random_state=42)
    preprocessor = ColumnTransformer([('numeric', num_pipe(),["Temperatur"]),("categoric", cat_pipe(encoder='onehot'),["Tenggorokan", "Badan", "Paru-Paru",]),])
    pipeline = Pipeline([('prep', preprocessor),('algo',GaussianNB())])
    pipeline.fit(X_train,y_train)
    
    st.set_option('deprecation.showPyplotGlobalUse', False)
    fig = plt.show()
    st.pyplot(fig)


def ShowTree():
    st.write('')
    st.subheader('Plot Confussion Matrix')
    X = df.drop(columns="Indikasi")
    y = df.Indikasi
    X_train, X_test , y_train, y_test = train_test_split(X,y, test_size=0.2 ,random_state=42)
    preprocessor = ColumnTransformer([('numeric', num_pipe(),["Temperatur"]),
    ("categoric", cat_pipe(encoder='onehot'),["Tenggorokan", "Badan", "Paru-Paru",]),])
    pipeline = Pipeline([('prep', preprocessor),('algo',tree.DecisionTreeClassifier())])
    pipeline.fit(X_train,y_train)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    fig = plt.show()
    st.pyplot(fig)

    
def Prediksi(tenggorokan,temperatur,badan,paru):
    X = df.drop(columns="Indikasi")
    y = df.Indikasi
    X_train, X_test , y_train, y_test = train_test_split(X,y, test_size=0.2 ,random_state=42)
    df_tes = pd.DataFrame(
     {'Tenggorokan' : [tenggorokan],
      'Temperatur'  : [temperatur],
      'Badan'       : [badan],
      'Paru-Paru'   : [paru]
     }
    )
    preprocessor = ColumnTransformer([('numeric', num_pipe(),["Temperatur"]),
    ("categoric", cat_pipe(encoder='onehot'),["Tenggorokan", "Badan", "Paru-Paru",]),])
    pipeline = Pipeline([('prep', preprocessor),('algo',GaussianNB())])
    testing = df_tes
    pipeline.fit(X_train,y_train)
    testing['Indikasi'] = pipeline.predict(testing)
    st.write(testing)
   

def display_results(X_train,X_pred):
    st.subheader('Data Training')
    st.write(X_train)
    
    st.subheader('Data Testing')
    st.write(X_pred)


    



def DataTesting():
    DataTraining()
    X = df.drop(columns="Indikasi")
    y = df.Indikasi
    X_train, X_test , y_train, y_test = train_test_split(X,y, test_size=0.2 ,random_state=42)

    #Cek Semua Data
    #print(df)
    #print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    preprocessor = ColumnTransformer([('numeric', num_pipe(),["Temperatur"]),
                                  ("categoric", cat_pipe(encoder='onehot'),
                                   ["Tenggorokan", "Badan", "Paru-Paru",]),
                                  ])

    pipeline = Pipeline([('prep', preprocessor),('algo',GaussianNB())])
    pipeline.fit(X_train,y_train)
    X_pred['Indikasi'] = pipeline.predict(X_pred)
    display_results(X_train,X_pred)
    ShowNB()
    
    
    
if __name__=="__main__":
   
    df = pd.read_excel('Covid19.xlsx')
    X_pred = pd.read_excel('xpredik.xlsx')

    st.sidebar.subheader('Data Training')
    file = st.sidebar.file_uploader(label='Pilih data training', type=('xlsx'))
      
    if file is not None:
        st.sidebar.write('File Uploaded')
        try:
            df_train = pd.read_excel(file)
            df = df_train
            
        except Exception as e:
            print(e)
            df = pd.read_excel('Covid19.xlsx')
    
    
    st.header("Indikasi Virus Covid-19 dengan Niave Bayes")
   
    if (st.button('Testing dengan NB', key=3)):
        st.write('Data Testing Menggunakan Naive Bayes')
        DataTesting()
    elif (st.button('Testing dengan Tree', key=4)):
        st.write('Data Testing Menggunakan Dicision Tree')
        DataTestingTree()
    else:
        st.write('')
    st.subheader('Data Prediksi')
    form = st.form(key='my-form')
    tenggorokan = form.text_input('kondisi tenggorokan(Normal/Sakit)', '-')
    temperatur = form.text_input('temperatur', '-')
    badan = form.text_input('kondisi badan(Fit/Lemas)', '-')
    paru = form.text_input('kondisi paru-paru(Sesal/Lega)', '-')
   
    submitNB = form.form_submit_button('Submit with NB')
    submitTree = form.form_submit_button('Submit with Tree')
    
    if submitNB:
        st.write('Prediksi Menggunakan Naive Bayes')
        Prediksi(tenggorokan,temperatur,badan,paru)
    elif submitTree:
        st.write('Prediksi Menggunakan Dicision Tree')
        prediksi_tree(tenggorokan,temperatur,badan,paru)
    else:
        st.write('Silakan Prediksi Data')

  
   
   







                                                                                                   
