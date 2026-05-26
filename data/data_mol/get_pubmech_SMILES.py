# pip install pubchempy pandas
import pubchempy as pcp
import pandas as pd

# Votre liste issue du NCI ou d'ailleurs
cancer_drugs = ["Doxorubicin", "Paclitaxel", "Imatinib", "Cisplatin"]
drug_data = []

print("Extraction des données depuis PubChem...")
for drug in cancer_drugs:
    try:
        # Recherche du composé par son nom
        compounds = pcp.get_compounds(drug, 'name')
        if compounds:
            c = compounds[0] # Prendre le meilleur résultat
            drug_data.append({
                'Name': drug,
                'CID': c.cid,
                'SMILES': c.isomeric_smiles,
                'Molecular_Weight': c.molecular_weight,
                'LogP': c.xlogp # Utile pour la perméabilité cellulaire
            })
            print(f"Trouvé : {drug}")
    except Exception as e:
        print(f"Erreur pour {drug}: {e}")

# Sauvegarde pour votre analyse en aval
df = pd.DataFrame(drug_data)
df.to_csv("cancer_drugs_pubchem.csv", index=False)
print("Terminé ! Fichier sauvegardé.")