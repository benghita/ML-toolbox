import pandas_profiling
import streamlit as st
from streamlit_pandas_profiling import st_profile_report
import pandas as pd
from ml_toolbox.AutoML import autoML
from io import StringIO
import numpy as np

# initialization of page variabal
if 'page' not in st.session_state:
    st.session_state.page = 0

def page1():
    st.session_state.page = 1

def page2():
    st.session_state.page = 2

def page3():
    st.session_state.page = 3

def main():

    st.title("Machine Learning Toolbox")

    placeholder = st.empty()
    
    if st.session_state.page == 0 : 

        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        if uploaded_file is not None:
         
            dataframe = pd.read_csv(uploaded_file)

            st.session_state.autom = autoML(dataframe)
            st.dataframe(st.session_state.autom.df)


            # Capture the output of df.info() in a StringIO buffer
            buffer = StringIO()
            st.session_state.autom.df.info(buf=buffer)
            info_str = buffer.getvalue()

            # Display the DataFrame info in Streamlit
            st.write("DataFrame Info:")
            st.code(info_str, language='text')

            st.button("Next", on_click = page1 )

    if st.session_state.page == 1 :
        with st.form("data_pre"):
                
            st.text('Data Preprocessing')
                        # View all key:value pairs in the session state
            s = []
            for k, v in st.session_state.items():
                s.append(f"{k}: {v}")
            st.write(s)
            st.dataframe(st.session_state.autom.df)
            # Hundle missing data
            st.write('Hunddle missing data : ')
            st.selectbox('Select the method of imputation for numirical values :',
            ('mean', 'median', 'most_frequent', 'constant'), key = 'num_impute')

            st.number_input('Insert the constant number :', key = 'number')
            
            st.selectbox('Select the method of imputation for catrgorical values :',
            ('most_frequent', 'constant'), key = 'ctg_impute')
            st.text_input('Insert the constant category :', key ='category')

            # Hundle data types 
            availble_cols = st.session_state.autom.df.columns.tolist()
            st.write('hundling data types : ')
            
            st.multiselect('Select columns to covert there types to number :', availble_cols, key = 'type_to_num')

            st.multiselect('Select columns to covert there types to category :', availble_cols, key ='type_to_ctg')

            st.multiselect('Select columns to covert there types to date :', availble_cols, key = 'type_to_dt')

            st.multiselect('Select columns to delet:', availble_cols, key = 'type_to_drp')

            st.form_submit_button("Apply changes", on_click = page2)
                

    if st.session_state.page == 2 :
        with st.form("data_encod"):
            #perform onehot encoding
# View all key:value pairs in the session state
            s = []
            for k, v in st.session_state.items():
                s.append(f"{k}: {v}")
            st.write(s)
            st.session_state.autom.handle_missing_and_types(st.session_state.num_impute, st.session_state.ctg_impute,
                                                st.session_state.number, st.session_state.category,
                                                st.session_state.type_to_num, st.session_state.type_to_ctg,
                                                st.session_state.type_to_dt, st.session_state.type_to_drp)

            st.title('Encoding : ')
            st.dataframe(st.session_state.autom.df)
            ctg_cols = st.session_state.autom.df.select_dtypes(include=['object']).columns.tolist()
            st.multiselect('Select categorical columns to apply one hot encoding :', ctg_cols, key = 'one_hot')
            st.slider('Select max encoding :', 1, 25, 10, key = 'max_encod')
            st.multiselect('Select categorical columns to apply ordinal encoding :', ctg_cols, key = 'ordinal_cols')
            numerical_columns = st.session_state.autom.df.select_dtypes(include=np.number).columns.tolist()
            st.multiselect('Select numerical columns to apply normalization :', numerical_columns, key = 'norm_cols')
            st.selectbox('select the method of normalization :',
                ('zscore', 'minmax', 'robust'), key = 'norm_method') 

            st.form_submit_button("Apply changes", on_click = page3) 
                

    if st.session_state.page == 3 :
        
        st.session_state.autom.handle_encoding_and_normalization(st.session_state.one_hot, st.session_state.max_encod,
                                                                st.session_state.ordinal_cols, st.session_state.norm_method,
                                                                st.session_state.norm_cols)
        st.dataframe(st.session_state.autom.df)
        pr = st.session_state.autom.df.profile_report()

        st_profile_report(pr)
        #f1, f3 = st.session_state.autom.visualize_data()
        #tab1, tab3 = st.tabs(["Correlation matrix", "Heatmap"])
        #with tab1:
        #    st.plotly_chart(f1, theme="streamlit", use_container_width=True)
        ##with tab2:
        ##    st.plotly_chart(f2, theme="streamlit", use_container_width=True)
        #with tab3:
        #    st.plotly_chart(f3, theme="streamlit", use_container_width=True)

    #if st.session_state.page == 4 :

if __name__ == "__main__":
    main()