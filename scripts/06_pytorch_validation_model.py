import pandas as pd
import warnings
import sys

pd.set_option('display.max_rows', 100)
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)
pd.set_option('future.no_silent_downcasting', True)



###############################################################################
# MAIN ########################################################################
###############################################################################
def main(file_id, data_path):
    df = pd.read_csv(data_path + file_id + '_valid_with_predictions.csv')

    df['ecart'] = df['ecoscore_score'] - df['predictions']
    moyenne_ecart = df['ecart'].mean()
    mediane_ecart = df['ecart'].median()
    print(f"moyenne_pred_valid: {moyenne_ecart}")
    print(f"median_pred_valid: {mediane_ecart}")

    def assign_tag(value):
        if 0 <= value < 20:
            return 'e'
        elif 20 <= value < 40:
            return 'd'
        elif 40 <= value < 60:
            return 'c'
        elif 60 <= value < 80:
            return 'b'
        elif 80 <= value <= 100:
            return 'a'
        else:
            return None  
    df['tags_pred'] = df['predictions'].apply(assign_tag)
    colonnes_a_afficher = ['ecoscore_score', 'predictions', 'ecoscore_tags', 'tags_pred']
    print(df[colonnes_a_afficher])

    def pourcentage_lettres_non_identiques(str1, str2):
        if str1 is None:
            str1 = ''
        if str2 is None:
            str2 = ''
        
        max_len = max(len(str1), len(str2))
        str1 = str1.ljust(max_len)
        str2 = str2.ljust(max_len)
        non_identiques = sum(c1 != c2 for c1, c2 in zip(str1, str2))
        pourcentage = (non_identiques / max_len) * 100 if max_len > 0 else 0 
        return pourcentage

    print(df[['ecoscore_tags', 'tags_pred']])
    df['pourcentage_non_identiques'] = df.apply(
        lambda row: pourcentage_lettres_non_identiques(row['ecoscore_tags'], row['tags_pred']),
        axis=1
    )
    lignes_non_identiques = df[df['pourcentage_non_identiques'] > 0]

    print("lignes avec des lettres non identiques:")
    print(lignes_non_identiques[['ecoscore_tags', 'tags_pred', 'pourcentage_non_identiques']])
    pourcentage_moyen_non_identiques = df['pourcentage_non_identiques'].mean()
    print(f"% non identique moyen: {pourcentage_moyen_non_identiques:.2f}%")

if __name__ == "__main__":
    file_id = sys.argv[1]
    data_path = sys.argv[2]
    main(file_id, data_path)