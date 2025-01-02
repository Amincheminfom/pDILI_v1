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

# URLs for the datasets
train_url = "https://github.com/Amincheminfom/pDILI_v1/raw/main/1_train_pDILI.csv"
test_url = "https://github.com/Amincheminfom/pDILI_v1/raw/main/2_test_pDILI.csv"

# Use the raw URL of the logo
logo_url = "https://raw.githubusercontent.com/Amincheminfom/pDILI_v1/main/pDILI_logo.jpg"

# Set the page config
st.set_page_config(
    page_title="pDILI_v1: Predict the Hepatotoxic Property of a Molecule",
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
    st.sidebar.success("Datasets loaded successfully from GitHub.")
except Exception as e:
    st.sidebar.error(f"Failed to load datasets: {e}")

# Descriptor columns
descriptor_columns = ['n7HRing', 'nS', 'n6aHRing', 'SlogP_VSA11',
       'NddssS', 'nG12FHRing', 'SlogP_VSA7', 'n7Ring', 'SMR_VSA6', 'GATS2pe',
       'SLogP', 'n10FaRing', 'AATSC0pe', 'Xch-6dv', 'VSA_EState6',
       'nBridgehead', 'JGI6', 'AATS2Z', 'ATSC6v', 'NddsN', 'ATSC6dv',
       'EState_VSA9', 'Xch-6d', 'AATSC2v', 'nCl', 'AATSC0i', 'JGI10', 'NdS',
       'JGT10', 'ATSC7v', 'VSA_EState7', 'FilterItLogS', 'n5Ring', 'Xc-5d',
       'EState_VSA2', 'AATS1p', 'NaaCH', 'AATSC0d', 'GATS1pe', 'ATSC7p',
       'Xc-5dv', 'C1SP3', 'NssssN', 'ATSC5p', 'AATS0v', 'n7aRing', 'n10FARing',
       'nRing', 'ATSC2v', 'n10FRing', 'GATS1Z', 'Xch-3d', 'NdsN', 'nG12Ring',
       'CIC0', 'ATSC4Z', 'GATS2d', 'nAromAtom', 'ATSC2dv', 'fMF', 'ATSC8dv',
       'nFAHRing']
# Train the model
try:
    X_train, y_train = train_data[descriptor_columns], train_data['Class']
    X_test, y_test = test_data[descriptor_columns], test_data['Class']

    model = RandomForestClassifier(
        n_estimators=50, max_depth=7, min_samples_split=10,
        min_samples_leaf=9, random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluate the model
    y_test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    st.sidebar.success(f"Model trained successfully! Testing accuracy: {test_accuracy:.4f}")

except Exception as e:
    st.sidebar.error(f"Error during training: {e}")


# Function to calculate LogP and Molecular Weight using RDKit
def calculate_logP_mw(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        logP = Descriptors.MolLogP(mol)  # LogP
        mw = Descriptors.MolWt(mol)  # Molecular Weight
        return logP, mw
    else:
        return None, None


# Function to generate 2D structure image from SMILES
def generate_2d_image(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        img = Draw.MolToImage(mol)
        return img
    else:
        return None

# Function to generate 2D structure image from SMILES with custom size and quality
def generate_2d_image(smiles, img_size=(300, 300)):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        img = Draw.MolToImage(mol, size=img_size, kekulize=True)  # Increase image resolution
        return img
    else:
        return None

# App title
st.title("pDILI_v1: Predict the Hepatotoxic Property of a Molecule")

about_expander = st.expander("What is pDILI_v1?", expanded=True)
with about_expander:
    st.write('''pDILI_v1 is a python package that allows users to predict the hepatotoxicity of a small molecule (1 = Toxic, 0 = Non-toxic) and also visualize the molecule.
               You can read about the theoretical background on pDILI_v1 in our paper. You can find our paper at the following link: [Paper](https://www.scopus.com/authid/detail.uri?authorId=57190176332).''')

# Input and prediction
smiles_input = st.text_input("Enter the SMILES string of a molecule:", "")
if smiles_input:
    mol = Chem.MolFromSmiles(smiles_input)
    if mol:
        col1, col2 = st.columns(2)

        # 2D Structure Visualization
        with col1:
            img = generate_2d_image(smiles_input)
            st.image(img, caption="2D Structure", use_container_width=False, width=300)

        # Prediction and Applicability Domain
        with col2:
            descriptors_df = pd.DataFrame([calc(mol)])
            descriptors_df.columns = [str(desc) for desc in calc.descriptors]

            # Ensure required descriptors are available
            available_columns = [col for col in descriptor_columns if col in descriptors_df.columns]
            missing_columns = [col for col in descriptor_columns if col not in descriptors_df.columns]
            if missing_columns:
                st.warning(f"Missing descriptors: {missing_columns}")

            descriptors_df = descriptors_df[available_columns]
            prediction = model.predict(descriptors_df)[0]
            st.success(f"Prediction: {'Toxic' if prediction == 1 else 'Non-toxic'}")

            # Leverage computation
            try:
                X_combined = np.vstack((X_train, descriptors_df.to_numpy()))
                leverage_matrix = np.dot(
                    X_combined,
                    np.linalg.pinv(np.dot(X_combined.T, X_combined))
                ).dot(X_combined.T)
                external_leverage = np.diag(leverage_matrix)[len(X_train):]
                leverage_threshold = 3 * X_train.shape[1] / X_train.shape[0]

                in_ad = external_leverage[0] <= leverage_threshold
                st.info(f"Inside Applicability Domain: {'Yes' if in_ad else 'No'}")

                # Plot the AD
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.axhline(y=leverage_threshold, color='r', linestyle='--', label='Leverage Threshold')
                ax.scatter(range(len(X_train)), np.diag(leverage_matrix)[:len(X_train)], label='Training Set Leverage',
                           color='blue', s=10)
                ax.scatter(len(X_train), external_leverage[0], label='Input Molecule', color='green', s=100,
                           edgecolor='black')
                ax.set_xlabel('Samples')
                ax.set_ylabel('Leverage')
                ax.set_title('Applicability Domain (AD) Plot')
                ax.legend(loc='upper right')
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error computing leverage: {e}")
    else:
        st.error("Invalid SMILES string.")
else:
    st.info("Please enter a SMILES string to predict its hepatotoxicity.")

# Contact information
contacts = st.expander("Contact", expanded=False)
with contacts:
    st.write('''#### Report an Issue
                 You are welcome to report a bug or contribute to the web 
                 application by filing an issue on [Github](https://github.com/Amincheminfom). 

                 #### Contact
                 For any question you can contact us through email:
                 - [Dr. Amin](mailto:pharmacist.amin@gmail.com)
                 - [Dr. Kar](mailto:pharmacist.amin@gmail.com)''')
