import streamlit as st
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas_summary as ps
import sweetviz as sv
import codecs
from pandas_profiling import ProfileReport 
import streamlit.components.v1 as components
from streamlit_pandas_profiling import st_profile_report



# Customm Funtion

def st_display_sweetviz(report_html,width=1000,height=500):
    report_file = codecs.open(report_html,'r')
    page = report_file.read()
    components.html(page,width=width,height=height,scrolling=True)


st.title("Machine Learning App")

html_temp = """
	<div style="background-color:tomato;"><p style="color:white;font-size:60px;">Data Anlysis</p></div>
	"""
st.markdown(html_temp,unsafe_allow_html=True)

#Remove side bar and made with streamlit
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

def main():
    """ML App --- """
    activites = ["EDA","Sweetviz EDA", "EDA Report","About"]
    choice = st.sidebar.selectbox("Select Activites",activites)
    
    if choice == "EDA":
        st.subheader("EDA")
        def file_selector(folder_path='./datasets'):
             filenames = os.listdir(folder_path)
             selected_filename = st.selectbox('Select a file', filenames)
             return os.path.join(folder_path, selected_filename)
	         
        filename = file_selector()
        st.write('You selected `%s`' % filename)
        df = pd.read_csv(filename) 
    
        #read a csv
        if st.checkbox("Show Data"):
            number = st.number_input("No of Rows to View",1,2000)
            st.dataframe(df.head(number))
            
        #show columns name
        if st.button("columns name"):
            st.success(df.columns)
            
        #show the shape of dataset
        if st.checkbox("Show Shape"):
            st.warning(df.shape)
        #show Dimention of Data set
        data_dim = st.radio("Show Dimentions By",("Rows","Columns"))
        if data_dim == "Rows":
            st.text("No of Rows")
            st.write(df.shape[0])
        elif data_dim == "Columns":
            st.text("No of Columns")
            st.write(df.shape[1])
        #show columns to select
        if st.checkbox("Select columns to show"):
            all_columns = df.columns.to_list()
            select_columns = st.multiselect("Select",all_columns)
            new_df = df[select_columns]
            st.dataframe(new_df)
        #Data Types
        if st.button("Data Types"):
            st.write(df.dtypes)
        # Summary of A Data
        if st.checkbox("Summary"):
            st.write(df.describe())
        #value count 
        if st.button("Value Count"):
            st.text("value counts By Target/Class")
            st.write(df.iloc[:,-1].value_counts())
        #summary of a column
        if st.checkbox("summary of a column"):
            dfs = ps.DataFrameSummary(df)
            all_columns = df.columns.to_list()
            select_column = st.selectbox("Select",all_columns)
            new_df = dfs[select_column]
            st.write(new_df)
            st.pyplot()
        #column Data Type 
        if st.button("Column Data Type"):
            st.write(dfs.columns_types)
        
        #visualization of a data
        st.header("Data Visualization")
        #correlation plot
        if st.checkbox("Correlation plot [matplotlib]"):
            plt.imshow(df.corr(),cmap="viridis")
            plt.colorbar()
            st.pyplot()
        # correlation plot using seaborn
        if st.checkbox("Correlation plot [Seaborn]"):
            plt.figure(figsize=(16,12))
            sns.heatmap(df.corr(),annot=True)
            st.pyplot()
        # correlation plot using plotly
        if st.checkbox("Correlation plot [Plotly]"):
            fig = st.imshow(df.corr())
            fig.show()
        # Drow count plot 
        if st.checkbox("Count plot"):
            st.text("Value count By Target")
            all_columns_name = df.columns.tolist()
            primary_col = st.selectbox("Select Primary column to group By",all_columns_name)
            select_column_name = st.multiselect("Select Columns",all_columns_name)
            if st.button("Plot"):
                st.text("Generating plot for {} and {}".format(primary_col,select_column_name))
                vc_plot = df.groupby(primary_col)[select_column_name].count()
            else:
                vc_plot = df.iloc[:,-1].value_counts()
            st.write(vc_plot.plot(kind="bar"))
            plt.show()
            st.pyplot()
        #Drow A pie plot
        if st.checkbox("Pie Plot"):
            st.text("pie plot of The Data")
            all_columns_name = df.columns.tolist()
            primary_col = st.selectbox("Select Column To Drow a Chart",all_columns_name)
            st.write(df[primary_col].value_counts().plot.pie(autopct= "%1.2f%%"))
            st.pyplot()
        # Drow a histogram 
        if st.checkbox("Hisogram"):
            st.text("Histogram  of The Data")
            all_columns_name = df.columns.tolist()
            primary_col = st.selectbox("Select Column To Drow a Hist plot",all_columns_name)
            plot = df[primary_col].plot.hist()
            if st.button("Generate Plot"):
                st.write(plot)
                st.pyplot()
            
            
        #violin plot 
        if st.checkbox("violin plot"):
            st.text("Violin plot of a Data")
            all_columns = df.columns.tolist()
            primary_col = st.selectbox("please Select a x-column To Drow a Violin plot",all_columns)
            sec_col = st.selectbox("please Select a y-column To Drow a Violin plot",all_columns)
            violin_plot = sns.catplot(x=primary_col,y=sec_col,kind="violin",data=df)
            if st.button("Generate Plot"):
                st.write(violin_plot)
                st.pyplot()
        #custom plots
        st.header("Custom plots")
        all_column_name = df.columns.tolist()
        plot_type = st.selectbox("Select the Plot Type",["area","bar","line","hist","box","kde"])
        select_column = st.multiselect("select a column to Drow",all_column_name)
        
        if st.button("Generate plot"):
            st.success("Generate a {} plot for {}".format(plot_type,select_column))
            if plot_type == "area":
                st.area_chart(df[select_column])
                st.pyplot()
            elif plot_type == "bar":
                st.bar_chart(df[select_column])
                st.pyplot()
            elif plot_type == "line":
                st.line_chart(df[select_column])
                st.pyplot()
            elif plot_type == "hist":
                st.write(df[select_column].plot(kind=plot_type))
                st.pyplot()
            elif plot_type == "box":
                st.write(df[select_column].plot(kind=plot_type))
                st.pyplot()
            elif plot_type == "kde":
                st.write(df[select_column].plot(kind=plot_type))
                st.pyplot()
    
           
            
    # Sweetviz EDA        
        
        
    elif choice == "Sweetviz EDA":
        st.subheader("Quick Analyise")
        data_file = st.file_uploader("Upload CSV",type=['csv'])
        if data_file is not None:
            df = pd.read_csv(data_file)
            st.dataframe(df.head())
            if st.button("Generate Sweetviz Report"):
                report = sv.analyze(df)
                report.show_html()
                st_display_sweetviz("SWEETVIZ_REPORT.html")

				
        
        
        

            
        
        
        
        
 # EDA Report       
        
        
    elif choice == "EDA Report":
        data_file1 = st.file_uploader("Upload CSV",type=['csv'])
        if data_file1 is not None:
            df = pd.read_csv(data_file1)
            st.dataframe(df.head())
            if st.button("Generate"):
                profile = ProfileReport(df)
                st_profile_report(profile)
        
        
        
        
        
        
        
    elif choice =="About":
        st.subheader("Made by Archit")
    
    
    
    
    
if __name__ == '__main__':
    main()
         
        
