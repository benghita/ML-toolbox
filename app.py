import streamlit as st
import pandas as pd
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report
from ml_toolbox.AutoML import autoML
from io import StringIO
import numpy as np
import pickle
import io

# initialization of page variabal
if 'page' not in st.session_state:
    st.session_state.page = 0

def next():
    st.session_state.page = st.session_state.page + 1

def pickle_model(model):
    """Pickle the model inside bytes. In our case, it is the "same" as 
    storing a file, but in RAM.
    """
    f = io.BytesIO()
    pickle.dump(model, f)
    return f


def main():

    if st.session_state.page == 0 : 
    # Page 1 : loading data and displaying its info

        st.header("Machine Learning Toolbox", divider='violet')
        st.markdown('This project is a simplified machine learning package inspired by PyCaret. \
                    It provides a user-friendly interface powered by Streamlit for both regression \
                    and classification tasks. The package is designed to automate common machine learning \
                    workflows and simplify the model selection process.')
        st.markdown('Upload your dataset : ')
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])

        if uploaded_file is not None:
         
            dataframe = pd.read_csv(uploaded_file)
            st.dataframe(dataframe)

            # Capture the output of df.info() in a StringIO buffer
            buffer = StringIO()
            dataframe.info(buf=buffer)
            info_str = buffer.getvalue()

            # Display the DataFrame info in Streamlit
            st.write("DataFrame Info:")
            st.code(info_str, language='text')
            
            # Select the target column (y)
            availble_cols = dataframe.columns.tolist()
            st.selectbox('Select the target column :', availble_cols, key = 'target')
            st.session_state.autom = autoML(dataframe, st.session_state.target)
            st.button("Next", on_click = next, use_container_width=True)

    if st.session_state.page == 1 :
    # Page 2 : Data preprocessing 
    
        with st.form("data_pre"):
                
            st.header('Data Preprocessing', divider='violet')

            st.dataframe(st.session_state.autom.df)

            # Handle missing data
            st.subheader('Handling missing data : ')
            st.selectbox('Choose the method of imputation for numerical values :',
            ('mean', 'median', 'most_frequent', 'constant'), key = 'num_impute')

            st.number_input('If constant insert a number :', key = 'number')
            
            st.selectbox('Choose the method of imputation for categorical values :',
            ('most_frequent', 'constant'), key = 'ctg_impute')
            st.text_input('If constant insert a value :', key ='category')

            st.divider()

            # Handle data types 
            availble_cols = st.session_state.autom.df.columns.tolist()
            features = [x for x in availble_cols if x != st.session_state.target]

            st.subheader('Handling data types : ')
            st.write('  Please ensure that you select the appropriate columns for each input section.')
            
            st.multiselect('Select columns to covert there types to number :', availble_cols, key = 'type_to_num')

            st.multiselect('Select columns to covert there types to category :', availble_cols, key ='type_to_ctg')

            #st.multiselect('Select columns to covert there types to date :', availble_cols, key = 'type_to_dt')
            
            st.multiselect('Select columns to delete:', features, key = 'type_to_drp')

            st.form_submit_button("Apply changes", on_click = next, use_container_width=True)
                

    if st.session_state.page == 2 :
    # Page 3 : Encoding, namalization and feature selection

        with st.form("data_encod"):
            st.session_state.autom.handle_missing_and_types(st.session_state.num_impute, st.session_state.ctg_impute,
                                                st.session_state.number, st.session_state.category,
                                                st.session_state.type_to_num, st.session_state.type_to_ctg, st.session_state.type_to_drp)

            st.header('Data Preprocessing', divider='violet')

            st.dataframe(st.session_state.autom.feautures)
            st.session_state.ctg_cols = st.session_state.autom.feautures.select_dtypes(include=['object']).columns.tolist()

            st.subheader('Encoding categorical data : ')
            #st.multiselect('Select categorical columns to apply one hot encoding :', ctg_cols, key = 'one_hot')
            st.slider('Select max encoding for one hot encoding:', 1, 25, 10, key = 'max_encod')
            #st.multiselect('Select categorical columns to apply ordinal encoding :', ctg_cols, key = 'ordinal_cols')
            #st.warning('Ensure that there are no shared columns between the ordinal encoding and one-hot encoding columns')
            #st.warning('Make sure that all categorical columns are encoded using either one-hot encoding or ordinal encoding to make them compatible with the models.')
            st.divider()

            st.subheader('Normalizing numerical data : ')
            numerical_columns = st.session_state.autom.feautures.select_dtypes(include=np.number).columns.tolist()
            st.multiselect('Select numerical columns to apply normalization :', numerical_columns, key = 'norm_cols')
            st.selectbox('Select the method of normalization :',('minmax', 'zscore', 'robust'), key = 'norm_method') 
            st.divider()
             
            st.subheader('Features selection : ')
            st.session_state.selection =  st.checkbox('Select the checkbox to enable feature selection and eliminate low variance features. ')
            st.selectbox('Choose the feature selection method : :',
                ('univariate', 'classic', 'sequential'), key = 'select_method') 
            st.number_input('Specify the threshold value for removing low variance features :', key = 'threshold')
            
            st.form_submit_button("Apply changes", on_click = next, use_container_width=True)  
                

    if st.session_state.page == 3 :
    # Page 4 : Generate report

        st.session_state.autom.handle_encoding_and_normalization(st.session_state.ctg_cols, st.session_state.max_encod,
                                                                st.session_state.norm_method,
                                                                st.session_state.norm_cols)
        if st.session_state.selection :
            st.session_state.autom.selection(st.session_state.select_method, st.session_state.threshold)

        st.subheader('Generating report : ')
        st.dataframe(st.session_state.autom.df)
        pr = st.session_state.autom.df.profile_report()
        with st.expander("REPORT", expanded=True):
            st_profile_report(pr)

        st.button("Start training", on_click = next, use_container_width=True )

    if st.session_state.page == 4 :
    # Page 5 : Train and evaluate the model

        next_model = False
        st.session_state.results = []

        task = st.session_state.autom.define_task()
        if task == 'regression' : 
            st.header('Train and evaluate regression models', divider='violet')
            st.subheader('Linear Regression : ')
            model_result = st.session_state.autom.linear_regression()
            st.write(model_result)
            st.session_state.results.append(model_result)
            next_model = True

        elif task == 'classification' : 
            st.header('Train and evaluate classification models', divider='violet')
            st.subheader('Logistic Regression : ')
            model_result = st.session_state.autom.logistic_regression()
            st.write(model_result)
            st.session_state.results.append(model_result)
            next_model = True

        else :
            st.warning('Can not identify the task')

        if next_model :
            # List of model names
            model_names = ['Ridge', 'Lasso', 'Decision Tree', 'Random Forest', 'Gradient Boosting', 'AdaBoost', 'SVR', 'KNN', 'MLP']
            
            # Loop through model names
            for model_name in model_names:
                st.subheader(f'{model_name} : ')
                try:
                    model_result = getattr(st.session_state.autom, model_name.lower().replace(' ', '_'))()
                    st.write(model_result)
                    st.session_state.results.append(model_result)
                except Exception as e:
                    st.write(f"An error occurred for {model_name}: {e}")

            st.button("Results Report", on_click = next, use_container_width=True )
            
    if st.session_state.page == 5 :
    # Page 6 : show the result and download models
        
        model_names = ['Regression', 'Ridge', 'Lasso', 'Decision Tree', 'Random Forest', 'Gradient Boosting', 'AdaBoost', 'SVR', 'KNN', 'MLP']
        st.header('Final Results : ', divider='violet')
        try : 
            df = pd.DataFrame(st.session_state.results, index=model_names)
            st.table(df)
        except Exception as e:
            st.warning('Missing values')

        best_model_name, best_model, best_scores = st.session_state.autom.best_model()
        st.write("Best model : ", best_model_name)
        st.write("Best model scores : ", best_scores)


        file_1 = pickle_model(best_model)
        if st.download_button("Download the best model as .pkl file", data=file_1, file_name="best-model.pkl"
                                , use_container_width=True) :
            st.success('Model downloaded successfully!')

        file_2 = pickle_model(st.session_state.autom.getmodels())
        if st.download_button("Download the trained models as .pkl file", data=file_2, file_name="trained_models.pkl"
                                , use_container_width=True) :
            st.success('Models downloaded successfully!')

        csv = st.session_state.autom.df.to_csv().encode('utf-8')
        st.download_button(
            label="Download the preprocessed dataset as CSV",
            data=csv,
            file_name='preprocessed_df.csv',
            mime='text/csv',
            use_container_width=True
        )
            
if __name__ == "__main__":
    main()