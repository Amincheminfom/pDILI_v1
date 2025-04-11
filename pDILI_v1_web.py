import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import Draw
from mordred import Calculator, descriptors
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import requests

# URLs for the datasets
train_url = "https://github.com/Amincheminfom/pDILI_v1/raw/main/1_train_pDILI.csv"
test_url = "https://github.com/Amincheminfom/pDILI_v1/raw/main/2_test_pDILI.csv"

# Use the raw URL of the logo
logo_url = "https://raw.githubusercontent.com/Amincheminfom/pDILI_v1/main/pDILI_logo.jpg"

# Set the page config
st.set_page_config(
    page_title="pDILI_v1: predictor of Drug-Induced Liver Injury of a molecule",
    layout="wide",
    page_icon=logo_url
)

model = None
train_data = None  # Initialize train_data globally
test_data = None

# Display the logo in the sidebar
st.sidebar.image(logo_url)

# Initialize Mordred calculator
calc = Calculator(descriptors, ignore_3D=True)

# Load datasets
try:
    train_data = pd.read_csv(train_url)
    test_data = pd.read_csv(test_url)
    st.sidebar.success("Datasets loaded successfully!")
except Exception as e:
    st.sidebar.error(f"Failed to load datasets: {e}")

# Descriptor columns
descriptor_columns = train_data.columns[1:-1].tolist()

# Train the model
try:
    X_train, y_train = train_data[descriptor_columns], train_data['Class']
    X_test, y_test = test_data[descriptor_columns], test_data['Class']

    model = RandomForestClassifier(
        n_estimators=50, max_depth=7, min_samples_split=10,
        min_samples_leaf=9, random_state=42
    )
    model.fit(X_train, y_train)

    # AUTHOR : Dr. Sk. Abdul Amin
    y_test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    st.sidebar.success(f"Thank you for using pDILI_v1!")

except Exception as e:
    st.sidebar.error(f"Error during training: {e}")

def generate_2d_image(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        img = Draw.MolToImage(mol)
        return img
    else:
        return None

def generate_2d_image(smiles, img_size=(300, 300)):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        img = Draw.MolToImage(mol, size=img_size, kekulize=True)  # Increase image resolution
        return img
    else:
        return None

# App title
st.title("pDILI_v1: predictor of Drug-Induced Liver Injury")

about_expander = st.expander("What is pDILI_v1?", expanded=True)
with about_expander:
    st.write('''pDILI_v1 is a python package that allows users to predict the association of drug-induced liver injury of a small molecule (1 = RISKy, 0 = Non-RISKy) and also visualize the molecule.
               You can read about the theoretical background on pDILI_v1 in our paper. You can find our paper at the following link: [Paper](https://doi.org/10.1021/acsomega.5c00075).''')

if 'descriptors_df' not in locals():
    # AUTHOR : Dr. Sk. Abdul Amin
    mol_placeholder = Chem.MolFromSmiles('c1ccccc1')
    descriptors_df = pd.DataFrame([calc(mol_placeholder)])
    descriptors_df.columns = [str(desc) for desc in calc.descriptors]
    descriptors_df = descriptors_df[descriptor_columns]  # AUTHOR : Dr. Sk. Abdul Amin

# Input and Prediction
smiles_input = st.text_input("Enter the SMILES string of a molecule:", "")
if smiles_input:
    mol = Chem.MolFromSmiles(smiles_input)
    if mol:
        col1, col2 = st.columns(2)

        # 2D Structure Visualization
        with col1:
            img = generate_2d_image(smiles_input)
            st.image(img, caption="2D Structure", use_container_width=False, width=300)

            # Calculate descriptors
            descriptors_df = pd.DataFrame([calc(mol)])
            descriptors_df.columns = [str(desc) for desc in calc.descriptors]

            available_columns = [col for col in descriptor_columns if col in descriptors_df.columns]
            missing_columns = [col for col in descriptor_columns if col not in descriptors_df.columns]
            if missing_columns:
                st.warning(f"Missing descriptors: {missing_columns}")

            if available_columns:
                descriptors_df = descriptors_df[available_columns]
                try:
                    # AUTHOR : Dr. Sk. Abdul Amin
                    prediction = model.predict(descriptors_df)[0]
                    st.success(f"Prediction: {'RISKy' if prediction == 1 else 'Non-RISKy'}")

                    X_combined = np.vstack((X_train, descriptors_df.to_numpy()))
                    leverage_matrix = np.dot(
                        X_combined,
                        np.linalg.pinv(np.dot(X_combined.T, X_combined))
                    ).dot(X_combined.T)
                    external_leverage = np.diag(leverage_matrix)[len(X_train):]
                    leverage_threshold = 3 * X_train.shape[1] / X_train.shape[0]

                    in_ad = external_leverage[0] <= leverage_threshold
                    st.info(f"Inside Applicability Domain: {'Yes' if in_ad else 'No'}")
                except Exception as e:
                    st.error(f"Error during prediction or leverage computation: {e}")

        # Prediction and Applicability Domain
        with col2:
            try:
                # Load the logo image
                response = requests.get(logo_url)
                logo_img = plt.imread(BytesIO(response.content), format="JPG")

                # Create AD plot
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.axhline(y=leverage_threshold, color='r', linestyle='--', label='Leverage Threshold')
                ax.scatter(range(len(X_test)), np.diag(leverage_matrix)[:len(X_test)],  # Test set leverage
                           color='blue', s=10)
                ax.scatter(len(X_test), external_leverage[0], label='Your Molecule', color='green', s=100,
                           edgecolor='black')
                ax.set_xlabel('Compound index')
                ax.set_ylabel('Leverage')
                ax.set_title('Applicability Domain (AD) Plot')
                ax.legend(loc='upper left')

                # Add logo below legend
                box = OffsetImage(logo_img, zoom=0.04)  # Adjust zoom for proper size
                ab = AnnotationBbox(box, (0.2, 0.7), xycoords="axes fraction", frameon=False)
                ax.add_artist(ab)

                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error computing leverage: {e}")
    else:
        st.error("Invalid SMILES string.")
else:
    st.info("Please enter a SMILES string to predict its hepatotoxicity.")

# Contact information AUTHOR : Dr. Sk. Abdul Amin
contacts = st.expander("Contact", expanded=False)
with contacts:
    st.write('''
            #### Report an Issue
                 
            You are welcome to report a bug or contribute to the web 
            application by filing an issue on [Github](https://github.com/Amincheminfom). 

            #### Contact
            For any question you can contact us through email:
                 
            - [Dr. S. Kar](mailto:skar@kean.edu)
            - [Dr. Sk. Abdul Amin](mailto:pharmacist.amin@gmail.com)
            ''')
