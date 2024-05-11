# Import relevant libraries 
import pandas as pd
import numpy as np
import streamlit as st
import pickle
import xgboost
# Load the model and the transformer 
with open("saved_steps.pkl","rb") as file:
    data = pickle.load(file)
Model = data["model"]
Transformer = data["Preprocessor"]

page = st.sidebar.radio("Menu", ["Homepage", "Make Prediction", "Learn More about Obesity"])
st.sidebar.title("Developer")
st.sidebar.image("jjj.jpeg", use_column_width=True)
st.sidebar.write("Rasheed Adeoti")
st.sidebar.write("https://www.linkedin.com/in/rasheed-adeoti/")

# Display content based on the selected page
if page == "Homepage":
    st.title("Obesity Level Prediction: Eating Habits & Fitness")
    st.image("kk.jpeg")
    st.markdown("""
    
    Obesity is a prevalent health concern affecting millions worldwide, with significant implications for overall well-being. This web application predicts your potential weight range based on an assessment of your dietary habits and physical activity levels. By evaluating various aspects of your eating routine, such as the frequency of high-calorie foods, vegetable consumption, meal frequency, snacking behaviors, water intake, and alcohol consumption, it provides predictions about your weight status.
    
    Additionally, the tool assesses your physical activity patterns, including your level of calorie monitoring, frequency of exercise, screen time habits, and mode of transportation. By considering these lifestyle factors alongside your dietary habits, it offers insights into your potential weight range.
    
    Personal information such as age, gender, height, and current weight is also taken into account to tailor the predictions to individual characteristics. This personalized approach enhances the accuracy of the weight range estimation, allowing users to gain valuable insights into their health and wellness journey.""")

    # Disclaimer
    # Displaying disclaimer
    st.subheader("Disclaimer")
    st.markdown("---")
    st.write("This application provides an estimate of your potential weight range and should not be considered a replacement for professional medical advice. It is advisable to consult with a healthcare professional for personalized guidance on weight management and overall health.")
    
elif page == "Make Prediction":
    st.markdown("<h1 style='text-align: center;'>Predict Obesity Level</h1>", unsafe_allow_html=True)
    with st.form("Obesity Level Prediction",clear_on_submit=True):
        col1, cols2, = st.columns(2)
        Height = col1.slider("Height",min_value=0.0, max_value=20.0,step=0.1)
        Age = cols2.slider("Age",min_value=0, max_value=200)
        Gender = col1.selectbox("Gender",options=["Male","Female"])
        FAVC = cols2.selectbox("Do you consume high caloric food frequently?",options=['yes', 'no'])
        FCVC = col1.slider("How frequent do you consume vegatables in your meal?",min_value=0, max_value=5)
        Weight = cols2.slider("Weight?",min_value=0.0, max_value=200.0,step=0.1)
        CAEC = col1.selectbox("How often do you consume food between meals?",options=('Sometimes', 'Frequently', 'Always', 'no'))  
        SMOKE = cols2.selectbox("Do you smoke?", options=["yes","no"])
        FAF = col1.slider("How frequently do you engage in physical exercise?",min_value=0, max_value=10)
        NCP = cols2.slider("How frequently do you consume main meals each day?",min_value=0, max_value=10)
        CALC = col1.selectbox("Do you consume alchol?",options=['no', 'Sometimes', 'Frequently', 'Always'])    
        SCC = cols2.selectbox("Do you track your daily calorie intake?",options=['yes', 'no'])
        CH2O =  col1.slider("How many liters of water do you consume daily on average?",min_value=0.0, max_value=20.0,step=0.1) 
        TUE = cols2.slider("How many hours do you spend using technology devices?",min_value=0.0, max_value=10.0,step=0.1)
        family_history_with_overweight	= col1.selectbox("Has a family member suffered or suffers from overweight?",options=['yes', 'no'])
        MTRANS = cols2.selectbox("Which method of transportation do you usually use?",options=['Public_Transportation', 'Walking', 'Automobile', 'Motorbike','Bike'])
        prediction = st.form_submit_button("Predict")
        if prediction:
            X = pd.DataFrame({'Gender':[Gender], 'Age':[Age], 'Height': [Height], 'Weight': [Weight], 'family_history_with_overweight': [family_history_with_overweight],'FAVC': [FAVC], 'FCVC':[FCVC], 'NCP':[NCP], 'CAEC':[CAEC], 'SMOKE':[SMOKE], 'CH2O':[CH2O], 'SCC':[SCC], 'FAF': [FAF], 'TUE':[TUE],'CALC':[CALC], 'MTRANS':[MTRANS]})
            print(X)
            X_trans = Transformer.transform(X)
            pred = Model.predict(X_trans)
            if pred[0] == 0:
                st.write(f"Your Obesity level is Insufficient Weight")
                st.write("Recommendation: Increasing calorie intake through nutrient-rich foods, such as lean proteins, healthy fats, and complex carbohydrates.")
            elif pred[0] == 1:
                st.write(f"Your Obesity level is Normal Weight")
                st.write("Recommendation: Increasing calorie intake through nutrient-rich foods, such as lean proteins, healthy fats, and complex carbohydrates.")
            elif pred[0] == 2:
                st.write(f"Your Obesity level is Obesity Type I")
                st.write("Recommendation:Adopt a balanced diet with portion control and regular exercise to promote weight loss.")
            elif pred[0] == 3:
                st.write(f"Your Obesity level is Obesity Type II")
                st.write("Recommendation:Adopt a balanced diet with portion control and regular exercise to promote weight loss.")
            elif pred[0] == 4:
                st.write(f"Your Obesity level is Obesity Type III")
                st.write("Recommedation: Consult a healthcare professional for personalized advice and potentially considering medical interventions or lifestyle modifications.")
            elif pred[0] == 5:
                st.write(f"Your Obesity level is Overweight_Level_I")
                st.write("Recommedation: Consult a healthcare professional for personalized advice and potentially considering medical interventions or lifestyle modifications.")
            else:
                st.write(f"Your Obesity level is Overweight Level II")
                st.write("Recommedation: Consult a healthcare professional for personalized advice and potentially considering medical interventions or lifestyle modifications.")
else:
    st.markdown(""" 
                ## Understanding Obesity
                * **Definition**: Obesity is a medical condition characterised by an excessive amount of body fat, which poses a risk to health. 
                The World Health Organization (WHO) identifies obesity as a leading preventable cause of death worldwide, impacting life
                expectancy negatively and increasing the incidence of health problems.
                * **How is obesity classified?**: Body mass index (BMI) is a calculation that takes a person’s weight and height into account to measure body size. 
                Doctors typically use it as a screening tool for obesity.
                """)
    table  = {'BMI': ['Less than 18.5', '18.5 to 24.9', '25.0 to 29.9','30.0 to 34.9','35.0 to 39.9','Higher than 40'],
        'Class': ['Under Weight', "Normal Weight", "Overweight", "Obesity Type I", "Obesity  Type II", "Obesity Type III"]
        }
    df = pd.DataFrame(table)
    st.table(df)
    st.markdown(""" 
                ## Causes of Obesity
                * **Diet**: The consumption of high-calorie foods, particularly those rich in sugars and fats, combined with large portion sizes, significantly contributes to the development of obesity. A diet that exceeds energy needs without sufficient physical activity leads to fat accumulation.
                * **Physical Inactivity**: A sedentary lifestyle, characterised by minimal physical activity, directly contributes to weight gain. Modern conveniences and technology have reduced the need for physical exertion in daily life, contributing to the obesity epidemic.
                * **Genetics**: Genetic predisposition plays a significant role in obesity. Individuals with a family history of obesity are at a higher risk, as genetics can influence fat storage and energy metabolism.
                * **Psychological Factors**: Emotional states such as stress and depression can lead to overeating as a coping mechanism, contributing to obesity. The relationship between emotions and eating behaviour is complex and multifaceted.
                * **Environmental Factors**: The environment, including access to healthy foods and safe areas for exercise, significantly affects lifestyle choices and obesity risk. Socioeconomic factors can influence diet and physical activity levels.
                * **Medicines**: Certain medications can lead to weight gain by altering the body's energy balance or increasing appetite. Medications for diabetes, depression, and high blood pressure are examples of those that can affect weight.
                """)
    st.markdown(""" 
                ## Health Risks Associated with Obesity
                Obesity significantly increases the risk for numerous health conditions that can affect nearly every system in the body:
                * **Cardiovascular Diseases**: Obesity contributes to heart disease and strokes, mainly through high blood pressure and abnormal cholesterol levels, posing serious risks to heart health.
                * **Metabolic Disorders**: Conditions such as type 2 diabetes and insulin resistance are closely linked to obesity, as excess body fat affects the body's ability to use insulin, leading to elevated blood sugar levels.
                * **Cancer**: There's a heightened risk for several types of cancer, including uterine, breast, colon, and liver cancer, among others, associated with obesity.
                * **Digestive Issues**: Obesity increases the likelihood of experiencing digestive problems like heartburn, gallbladder disease, and serious liver conditions, including fatty liver disease.
                * **Respiratory Problems**: Excess weight is a key factor in the development of sleep apnoea and can contribute to other respiratory issues, impacting overall respiratory health.
                * **Joint and Inflammation Issues**: Conditions such as osteoarthritis are more common in individuals with obesity due to the increased stress on weight-bearing joints and systemic inflammation.
                * **Severe COVID-19 Symptoms**: Individuals with obesity are at a higher risk for developing more severe complications if they contract COVID-19, including increased likelihood of hospitalisation, ICU admission, and mechanical ventilation.
                * **Other Health Concerns**: Obesity also increases the risk for dyslipidaemia, kidney disease, and complications related to pregnancy, fertility, and sexual function, in addition to mental health issues like depression and anxiety and challenges with physical functioning.
                """)
    st.subheader("Sources")
    st.markdown("---")
    st.markdown(""" 
                * HealthScope (https://healthscope.streamlit.app/)
                * Healthline (https://www.healthline.com/health/obesity#complications)
                """)
    


    
                

